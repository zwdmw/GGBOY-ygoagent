"""A ygopro LAN duel client that plays a seat with a pluggable policy.

The client joins a hosted room, submits a deck, and from ``STOC_DUEL_START``
onwards drives one duel: every ``STOC_GAME_MSG`` carries exactly one
ygopro-core message buffer (``SingleDuel::Analyze`` sends one message per
packet, ``gframe/single_duel.cpp:600+``), so there is no framing to guess.
Prompts become action lists via :mod:`.actions`, the policy picks an index, and
the encoded bytes go back as ``CTOS_RESPONSE``.

Two protocol details are easy to miss and stall the duel silently:

* the server ignores a packet whose type is not the one it is waiting for
  (``netserver.cpp:240``), so the handshake order matters;
* when the room has a time limit the server first expects ``CTOS_TIME_CONFIRM``
  and only then accepts the response (``single_duel.cpp`` ``WaitforResponse`` /
  ``TimeConfirm``).

There is no duel engine on this side: the board is reconstructed from the
message stream by :mod:`.board`, which is what the product's shadow duel does.
"""

from __future__ import annotations

import random
import socket
import threading
import time
from dataclasses import dataclass, field

from . import constants as C
from . import protocol as P
from .actions import (
    Reader,
    SelectContext,
    UnsupportedMessage,
    parse_select,
)
from .board import ShadowBoard

__all__ = ["DuelResult", "NetDuelClient", "DuelError"]

# prompts that need an answer from us
SELECT_MESSAGES = frozenset(
    {
        C.MSG_SELECT_BATTLECMD,
        C.MSG_SELECT_IDLECMD,
        C.MSG_SELECT_EFFECTYN,
        C.MSG_SELECT_YESNO,
        C.MSG_SELECT_OPTION,
        C.MSG_SELECT_CARD,
        C.MSG_SELECT_CHAIN,
        C.MSG_SELECT_PLACE,
        C.MSG_SELECT_POSITION,
        C.MSG_SELECT_TRIBUTE,
        C.MSG_SELECT_COUNTER,
        C.MSG_SELECT_SUM,
        C.MSG_SELECT_DISFIELD,
        C.MSG_SORT_CARD,
        C.MSG_SELECT_UNSELECT_CARD,
        C.MSG_ANNOUNCE_RACE,
        C.MSG_ANNOUNCE_ATTRIB,
        C.MSG_ANNOUNCE_CARD,
        C.MSG_ANNOUNCE_NUMBER,
        C.MSG_ROCK_PAPER_SCISSORS,
    }
)


class DuelError(RuntimeError):
    """The duel could not be played to the end."""


@dataclass
class DuelResult:
    our_player: int = -1  # duel-side player index (0 goes first)
    winner: int = -1  # duel-side player index, 2 = draw
    won: bool | None = None
    win_reason: int = -1
    lp: tuple[int, int] = (0, 0)
    turns: int = 0
    decisions: int = 0  # prompts the policy answered
    auto_responses: int = 0  # prompts answered without asking the policy
    forced_actions: int = 0  # single-option prompts taken without a decision
    seconds: float = 0.0
    went_first: bool | None = None
    error: str = ""
    trace: list = field(default_factory=list)

    def as_dict(self) -> dict:
        d = dict(self.__dict__)
        d.pop("trace", None)
        d["lp"] = list(self.lp)
        return d


class NetDuelClient:
    """One seat of one duel.

    ``policy`` is called as ``policy.choose(state)`` with a
    :class:`~mirrorforce.netduel.policy.DecisionState` and must return an index
    into ``state.actions``.
    """

    def __init__(
        self,
        host: str,
        port: int,
        name: str,
        main: list[int],
        extra: list[int],
        policy,
        side: list[int] | None = None,
        version: int = P.PRO_VERSION,
        seed: int = 0,
        go_first: bool | None = None,
        timeout: float = 120.0,
        max_options: int = 24,
        card_pool=None,
        track_board: bool = True,
        trace: bool = False,
        single_option_is_forced: bool = True,
        opponent_seed: int | None = None,
        capture: list | None = None,
        log=None,
    ):
        self.host = host
        self.port = port
        self.name = name
        self.main = list(main)
        self.extra = list(extra)
        self.side = list(side or [])
        self.policy = policy
        self.version = version
        self.rng = random.Random(seed)
        self.go_first = go_first
        self.timeout = timeout
        self.max_options = max_options
        self.card_pool = card_pool
        self.track_board = track_board
        self.trace = trace
        #: match ygoenv, which auto-takes a prompt that offers one option.
        #: Off only for the tests that assert on the old behaviour.
        self.single_option_is_forced = single_option_is_forced
        #: WINDBOT_SEED of the opponent, when we set one.  Its throw is the
        #: first draw off that seed, so we can answer with the hand that beats
        #: it and win the throw every time -- which is the only way the seat
        #: becomes ours to choose.
        self.opponent_seed = opponent_seed
        self.rps_wins = 0
        self.rps_throws = 0
        #: when set, every ``STOC_GAME_MSG`` body lands here as ``(msg, bytes)``.
        #: Replaying a captured stream through the shadow board offline is the
        #: only way to debug a tracking bug without re-running a live duel and
        #: hoping it happens again.
        # a policy may bring its own sink, which is how the alignment recorder
        # gets the stream without every caller having to thread it through
        self.capture = capture if capture is not None else getattr(policy, "capture", None)
        self.log = log or (lambda *a: None)

        self.stream: P.PacketStream | None = None
        self.host_info: P.HostInfo | None = None
        # the orchestrator waits on these before pressing "start duel"
        self.joined = threading.Event()
        self.started = threading.Event()
        # the host only starts the duel once both duelists are ready, and we
        # see the opponent's state through STOC_HS_PLAYER_CHANGE
        self.opponent_ready = threading.Event()
        self.lobby_pos = -1
        self.result = DuelResult()
        self.board = ShadowBoard()
        self.board_problems: list[str] = []
        self.ctx = SelectContext(max_options=max_options, rng=self.rng, card_pool=card_pool)
        self._lp = [8000, 8000]
        self._pending_time_confirm = False
        self._duel_over = False

    # -- connection --------------------------------------------------------

    def connect(self) -> None:
        sock = socket.create_connection((self.host, self.port), timeout=self.timeout)
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        self.stream = P.PacketStream(sock)
        self.stream.send(P.CTOS.PLAYER_INFO, P.player_info(self.name))
        self.stream.send(P.CTOS.JOIN_GAME, P.join_game(self.version))

    def surrender(self) -> None:
        """Concede cleanly.  Accepted from any state, unlike everything else.

        Dropping the socket instead would make the host award the win by
        disconnect and, if we were the host, tear the whole server down.
        """
        if self.stream is not None:
            try:
                self.stream.send(P.CTOS.SURRENDER)
            except OSError:
                pass

    def close(self) -> None:
        if self.stream is not None:
            try:
                self.stream.send(P.CTOS.LEAVE_GAME)
            except OSError:
                pass
            self.stream.close()
            self.stream = None

    # -- main loop ---------------------------------------------------------

    def run(self) -> DuelResult:
        start = time.monotonic()
        try:
            self.connect()
            while not self._duel_over:
                op, payload = self.stream.recv(self.timeout)
                self._handle(op, payload)
        except (DuelError, UnsupportedMessage) as exc:
            self.result.error = f"{type(exc).__name__}: {exc}"
            # concede so the host finishes the duel instead of waiting forever
            self.surrender()
        except (ConnectionError, socket.timeout, OSError) as exc:
            if not self._duel_over:
                self.result.error = f"{type(exc).__name__}: {exc}"
        finally:
            self.joined.set()
            self.started.set()
            self.opponent_ready.set()
            self.result.seconds = time.monotonic() - start
            self.result.lp = (self._lp[0], self._lp[1])
            try:
                self.policy.on_duel_end(self.result)
            except Exception as exc:  # tearing a policy down must not lose the duel
                self.log(f"policy teardown failed: {exc}")
            self.close()
        return self.result

    def _handle(self, op: int, payload: bytes) -> None:
        if op == P.STOC.JOIN_GAME:
            self.host_info = P.HostInfo.unpack(payload)
            self.log(f"joined: {self.host_info}")
            if self.host_info.mode != 0:
                raise DuelError(
                    f"room mode {self.host_info.mode} is not a single duel; "
                    "match and tag rooms need side decking support"
                )
            self.stream.send(P.CTOS.UPDATE_DECK, P.update_deck(self.main + self.extra, self.side))
        elif op == P.STOC.TYPE_CHANGE:
            self.lobby_pos = payload[0] & 0xF
            self.log(f"lobby position {self.lobby_pos} (raw 0x{payload[0]:02x})")
            if self.lobby_pos == P.NETPLAYER_TYPE_OBSERVER:
                # every duelist slot was taken; try to claim one
                self.stream.send(P.CTOS.HS_TODUELIST)
                return
            self.stream.send(P.CTOS.HS_READY)
            self.joined.set()
        elif op == P.STOC.HS_PLAYER_CHANGE:
            pos = payload[0] >> 4
            state = payload[0] & 0xF
            if pos != self.lobby_pos and state == P.PLAYERCHANGE.READY:
                self.opponent_ready.set()
            self.log(f"lobby: seat {pos} state 0x{state:x}")
        elif op == P.STOC.HS_PLAYER_ENTER:
            self.log(f"lobby: {P.decode_name(payload[:40])} took seat {payload[40]}")
        elif op == P.STOC.ERROR_MSG:
            msg = payload[0]
            code = int.from_bytes(payload[4:8], "little")
            raise DuelError(f"server error msg={msg} code=0x{code:x}")
        elif op == P.STOC.SELECT_HAND:
            self.rps_throws += 1
            if self.opponent_seed is not None and self.rps_throws == 1:
                from .dotnetrandom import hand_that_wins

                hand = hand_that_wins(self.opponent_seed)
            else:
                # a re-throw means the prediction was wrong (or no seed was
                # set); fall back rather than repeat a losing hand
                hand = self.rng.randint(1, 3)
            self.stream.send(P.CTOS.HAND_RESULT, P.hand_result(hand))
        elif op == P.STOC.HAND_RESULT:
            # only the winner is asked to choose, so this is how we learn
            # whether the prediction held
            pass
        elif op == P.STOC.SELECT_TP:
            self.rps_wins += 1
            first = self.rng.random() < 0.5 if self.go_first is None else self.go_first
            self.stream.send(P.CTOS.TP_RESULT, P.tp_result(1 if first else 0))
        elif op == P.STOC.TIME_LIMIT:
            # the clock is ours: the server parked us in CTOS_TIME_CONFIRM and
            # will silently drop a CTOS_RESPONSE until we confirm
            player = payload[0]
            if self.result.our_player < 0 or player == self.result.our_player:
                self.stream.send(P.CTOS.TIME_CONFIRM)
        elif op == P.STOC.GAME_MSG:
            self._game_msg(payload)
        elif op == P.STOC.DUEL_END:
            self._duel_over = True
        elif op in (P.STOC.CHANGE_SIDE, P.STOC.WAITING_SIDE):
            # match mode only; a single duel never gets here
            raise DuelError("side decking requested; only single duels are supported")
        # DECK_COUNT / DUEL_START / CHAT / HS_* / REPLAY need no answer

    # -- duel messages -----------------------------------------------------

    def _game_msg(self, payload: bytes) -> None:
        if not payload:
            return
        msg = payload[0]
        body = payload[1:]
        if self.capture is not None:
            self.capture.append((msg, bytes(body)))
        self._track(msg, body)
        if self.track_board:
            # board tracking is observation, never a reason to lose a duel: a
            # malformed or unmodelled buffer is counted, not raised
            try:
                self.board.apply(msg, body)
            except Exception as exc:
                self.board_problems.append(f"msg {msg}: board update failed: {exc}")
        # Stateful policies may need the same public message history a real
        # player saw. Deliver only STOC_GAME_MSG bytes received by this client;
        # never synthesize the opponent's private MSG_SELECT_* prompt from an
        # in-process core buffer.
        message_observer = getattr(self.policy, "observe_game_message", None)
        if message_observer is not None:
            message_observer(msg, body)
        if msg in SELECT_MESSAGES:
            self._answer(msg, body)

    def _track(self, msg: int, body: bytes) -> None:
        r = Reader(body)
        if msg == C.MSG_START:
            playertype = r.u8()
            self.result.our_player = playertype & 0xF
            self.result.went_first = self.result.our_player == 0
            r.u8()  # duel rule
            self._lp[0] = r.i32()
            self._lp[1] = r.i32()
            self.ctx.our_player = self.result.our_player
            self.board.start(self.result.our_player, self.main, self.extra)
            # a policy that keeps state per duel (an observation encoder, a
            # recurrent state) can only set it up once the seat is known
            binder = getattr(self.policy, "bind", None)
            if binder is not None:
                binder(self)
            self.log(f"duel start, we are player {self.result.our_player}")
            self.started.set()
        elif msg == C.MSG_NEW_TURN:
            self.result.turns += 1
            self.board.turn_player = r.u8()
        elif msg == C.MSG_NEW_PHASE:
            self.ctx.current_phase = r.u16()
            self.board.phase = self.ctx.current_phase
        elif msg == C.MSG_HINT:
            hint_type = r.u8()
            r.u8()
            value = r.u32()
            if hint_type == C.HINT_SELECTMSG and value == 501:
                self.ctx.discard_hand = True
        elif msg == C.MSG_DAMAGE:
            # every player byte on the wire is the duel-side index, which is
            # what we index _lp by, so no LocalPlayer mapping is needed
            player = r.u8()
            self._lp[player] = max(0, self._lp[player] - r.u32())
        elif msg == C.MSG_RECOVER:
            player = r.u8()
            self._lp[player] += r.u32()
        elif msg == C.MSG_PAY_LPCOST:
            player = r.u8()
            self._lp[player] = max(0, self._lp[player] - r.u32())
        elif msg == C.MSG_LPUPDATE:
            player = r.u8()
            self._lp[player] = r.u32()
        elif msg == C.MSG_WIN:
            winner = r.u8()
            self.result.winner = winner
            self.result.win_reason = r.u8()
            if winner > 1:
                self.result.won = None
            else:
                self.result.won = winner == self.result.our_player
            self._duel_over = True
        elif msg == C.MSG_RETRY:
            # the server re-arms us and waits again, but we no longer hold the
            # prompt that produced the bad answer, so concede rather than hang
            self.surrender()
            raise DuelError("engine rejected the response (MSG_RETRY)")

    def _answer(self, msg: int, body: bytes) -> None:
        result = parse_select(msg, body, self.ctx)
        if self.track_board and result.cards:
            try:
                problems = self.board.cross_check(result.cards)
            except Exception as exc:
                problems = [f"cross-check failed: {exc}"]
            for problem in problems:
                self.board_problems.append(f"msg {msg}: {problem}")
                self.log(f"shadow board disagrees with the engine: {problem}")
        if result.auto_response is not None:
            self.result.auto_responses += 1
            self._send_response(result.auto_response)
            return
        selector = result.selector
        while True:
            actions = selector.options()
            if not actions:
                raise DuelError(f"message {msg} offers no legal action")
            if self.single_option_is_forced and len(actions) == 1:
                # ygoenv never shows the agent a prompt with one option: it
                # takes it, records it in the action history and moves on
                # (``ygopro.h:3931-3941``).  Asking the policy here instead
                # would feed the recurrent state an extra step that training
                # never had, on top of a choice that does not exist -- so the
                # observation and the decision count would both drift from
                # what the checkpoint was trained against.
                idx = 0
                self.result.forced_actions += 1
                observer = getattr(self.policy, "observe", None)
                if observer is not None:
                    observer(msg, actions[0], self.result.turns, self.ctx.current_phase)
                data = selector.choose(idx)
                if data is not None:
                    self._send_response(data)
                    return
                continue
            state = self._decision_state(selector, actions)
            idx = int(self.policy.choose(state))
            if not 0 <= idx < len(actions):
                raise DuelError(
                    f"policy returned {idx} for {len(actions)} options ({msg})"
                )
            self.result.decisions += 1
            if self.trace:
                self.result.trace.append(
                    {
                        "msg": msg,
                        "turn": self.result.turns,
                        "n": len(actions),
                        "idx": idx,
                        "action": actions[idx].describe(),
                    }
                )
            data = selector.choose(idx)
            if data is not None:
                self._send_response(data)
                return

    def _decision_state(self, selector, actions):
        from .policy import DecisionState

        return DecisionState(
            msg=selector.msg,
            player=selector.player,
            our_player=self.result.our_player,
            actions=actions,
            board=self.board,
            turn=self.result.turns,
            phase=self.ctx.current_phase,
            lp=tuple(self._lp),
        )

    def _send_response(self, data: bytes) -> None:
        if len(data) > 255:
            raise DuelError(f"response of {len(data)} bytes exceeds the wire limit")
        tee = getattr(self, "response_tee", None)
        if tee is not None:
            # The live searcher replays the duel from these bytes.  Both
            # seats' clients share one list; the engine holds one prompt at
            # a time, so append order is engine consumption order.
            tee.append(bytes(data))
        self.stream.send(P.CTOS.RESPONSE, data)
