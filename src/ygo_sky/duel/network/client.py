"""Owner-started external duel client with tournament-aware reconnects."""
from itertools import combinations
from contextlib import suppress
import hashlib
import lzma
import secrets
import socket
import struct
import time

from .preference import prefer_first
from . import deck_sync
from .vendor.netduel import constants as C, protocol as P
from .vendor.netduel.actions import MultiSelector, Reader, parse_select, ls_to_spec, SelectContext
from .vendor.netduel.board import ShadowBoard
from .vendor.netduel.client import NetDuelClient, DuelError, DuelResult, SELECT_MESSAGES
from .vendor.replay.actions import MultiSelect


MAX_MATCH_GAMES = 10000
CONNECT_WINDOW_SECONDS = 10 * 60
RECONNECT_WINDOW_SECONDS = 5 * 60
TRANSITION_IDLE_SECONDS = 5 * 60
RECV_POLL_SECONDS = 5
RECONNECT_BACKOFF_SECONDS = (1, 2, 5, 10)


class ReconnectRequired(ConnectionError):
    """The transport is stale but the tournament task is still active."""


def parse_prompt(msg, body, context):
    if msg != C.MSG_SELECT_TRIBUTE:
        return parse_select(msg, body, context)
    r = Reader(body)
    player, cancelable, minimum, maximum, count = (r.u8() for _ in range(5))
    specs, codes, weights, cards = [], [], [], []
    for _ in range(count):
        code, controller, location, sequence, weight = r.u32(), r.u8(), r.u8(), r.u8(), r.u8()
        specs.append(ls_to_spec(location, sequence, 0, controller != player))
        codes.append(code)
        weights.append(weight)
        cards.append((code, controller, location, sequence))
    if r.left:
        raise DuelError("Trailing bytes in tribute prompt")
    combos = [list(row) for size in range(0 if minimum == 0 else 1, min(count, maximum) + 1)
              for row in combinations(range(count), size)
              if sum(weights[i] for i in row) >= minimum]
    if not combos:
        raise DuelError("No legal tribute combination")
    from .vendor.netduel.actions import SelectResult
    selector = MultiSelector(msg, player, MultiSelect(minimum, maximum, 0, specs, 1, combos), codes)
    return SelectResult(msg, player, selector=selector, cards=cards)


def legacy_message(msg, body):
    if msg == C.MSG_SELECT_CHAIN:
        if len(body) < 12 or len(body) != 12 + body[1] * 13:
            raise DuelError("Invalid legacy chain prompt")
        converted = bytearray(body[:3] + body[4:12])
        for offset in range(12, len(body), 13):
            converted.extend(body[offset:offset + 1] + body[3:4] +
                             body[offset + 1:offset + 13])
        return bytes(converted)
    if msg == C.MSG_CONFIRM_CARDS:
        if len(body) < 2 or len(body) != 2 + body[1] * 7:
            raise DuelError("Invalid legacy card disclosure")
        return body[:1] + b"\0" + body[1:]
    return body


class RecordingStream(P.PacketStream):
    def __init__(self, sock, journal):
        super().__init__(sock)
        self.journal = journal

    def send(self, opcode, payload=b""):
        redacted = opcode in (P.CTOS.JOIN_GAME, P.CTOS.CHAT)
        self.journal("out", opcode, b"" if redacted else payload, redacted=redacted)
        super().send(opcode, payload)

    def recv(self, timeout=None):
        opcode, payload = super().recv(timeout)
        # Server chat can echo authentication commands.
        self.journal("in", opcode, b"" if opcode == P.STOC.CHAT else payload,
                     redacted=opcode == P.STOC.CHAT)
        return opcode, payload


class ExternalClient(NetDuelClient):
    def __init__(self, *args, password="", preference="auto", journal=None,
                 state_change=None, save_replay=None, deck_source="local",
                 auth_mode="none", account_password="", game_finished=None,
                 deck_sync_protocol=None, require_deck_info=True, assigned_deck=None,
                 connect_window=CONNECT_WINDOW_SECONDS,
                 reconnect_window=RECONNECT_WINDOW_SECONDS,
                 transition_idle=TRANSITION_IDLE_SECONDS,
                 poll_interval=RECV_POLL_SECONDS,
                 reconnect_backoff=RECONNECT_BACKOFF_SECONDS, **kwargs):
        super().__init__(*args, **kwargs)
        if deck_source not in ("local", "server_random") or auth_mode not in ("none", "duelchronicle"):
            raise ValueError("Invalid external deck/account mode")
        if auth_mode == "duelchronicle" and (deck_source != "server_random" or not account_password):
            raise ValueError("DuelChronicle requires server-assigned decks and account authentication")
        limits = (connect_window, reconnect_window, transition_idle, poll_interval)
        if any(type(value) not in (int, float) or value <= 0 for value in limits):
            raise ValueError("Invalid external connection timeout")
        if (
            not isinstance(reconnect_backoff, (tuple, list))
            or not reconnect_backoff
            or any(type(value) not in (int, float) or value < 0 for value in reconnect_backoff)
        ):
            raise ValueError("Invalid external reconnect backoff")
        self.deck_source, self.auth_mode = deck_source, auth_mode
        if deck_sync_protocol not in (None, "relay-v1"):
            raise ValueError("deck_sync_unsupported_protocol")
        self.deck_sync_protocol = deck_sync_protocol
        self.sync_nonce = secrets.token_hex(16) if deck_sync_protocol else None
        self.require_deck_info = require_deck_info
        self.pending_deck = None
        self.deck_sync = {"state": "pending" if deck_source == "server_random" or deck_sync_protocol else "local"}
        self.assigned_deck = assigned_deck or (lambda document, deck: None)
        self.account_password = account_password
        self.auth_sent = False
        self.authenticated = auth_mode == "none"
        self.auth_deadline = None
        self.submitted = False
        self.match_mode = False
        self.match_done = False
        self.match_kill = False
        self.games = []
        self.game_finished = game_finished or (lambda game: None)
        self.own_ready = False
        if deck_source == "server_random" or deck_sync_protocol:
            from .server_deck import ServerDeckBoard
            self.main, self.extra, self.side = [], [], []
            self.board = ServerDeckBoard()
        self.go_first = (None if deck_source == "server_random" and preference == "auto"
                         else prefer_first(self.main, preference))
        self.password = password
        self.journal = journal or (lambda *a, **kw: None)
        self.state_change = state_change or (lambda *a, **kw: None)
        self.save_replay = save_replay or (lambda data: None)
        self.host_player = False
        self.replay_received = False
        self.replay_hashes = set()
        self.replay_errors = []
        self.initial_deck_counts = None
        self.terminal_seen = False
        self.terminal_time = None
        self.selector = None
        self.cancelled = False
        self.lifecycle_state = "idle"
        self.connect_window = float(connect_window)
        self.reconnect_window = float(reconnect_window)
        self.transition_idle = float(transition_idle)
        self.poll_interval = float(poll_interval)
        self.reconnect_backoff = tuple(float(value) for value in reconnect_backoff)
        self.last_packet_at = None
        self.ever_joined = False
        self.reconnect_started = None
        self.reconnect_attempts = 0
        self.resume_current_game = False
        self.ctx.known_codes = list(dict.fromkeys(self.main + self.extra + self.side))

    def change_state(self, state, **fields):
        self.lifecycle_state = state
        self.state_change(state, **fields)

    def reset_handshake(self):
        self.auth_sent = False
        self.authenticated = self.auth_mode == "none"
        self.auth_deadline = None
        self.submitted = False
        self.own_ready = False
        self.host_player = False
        self.lobby_pos = -1
        self.opponent_ready.clear()
        self.pending_deck = None

    def connect(self, reconnecting=False):
        self.reset_handshake()
        self.change_state(
            "reconnecting" if reconnecting else "connecting",
            attempt=self.reconnect_attempts if reconnecting else 0,
        )
        sock = socket.create_connection(
            (self.host, self.port),
            timeout=min(15, self.connect_window),
        )
        sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)
        for option, value in (
            ("TCP_KEEPIDLE", 60),
            ("TCP_KEEPINTVL", 15),
            ("TCP_KEEPCNT", 4),
        ):
            if hasattr(socket, option):
                with suppress(OSError):
                    sock.setsockopt(
                        socket.IPPROTO_TCP,
                        getattr(socket, option),
                        value,
                    )
        self.stream = RecordingStream(sock, self.journal)
        self.last_packet_at = time.monotonic()
        self.stream.send(P.CTOS.PLAYER_INFO, self.name.encode("utf-16-le").ljust(40, b"\0"))
        # UTF-16 code units, not Python code points, define this protocol field.
        encoded = self.password.encode("utf-16-le")
        if len(encoded) > 38:
            raise DuelError("Room password is too long")
        self.stream.send(P.CTOS.JOIN_GAME,
                         struct.pack("<H2xI", self.version, 0) + encoded.ljust(40, b"\0"))
        if self.deck_sync_protocol:
            self.stream.send(deck_sync.OPCODE, deck_sync.request(self.sync_nonce))

    def surrender(self):
        with suppress(Exception):
            super().surrender()

    def close(self):
        stream = self.stream
        try:
            with suppress(Exception):
                super().close()
        finally:
            if stream is not None:
                stream.close()
            self.stream = None

    def drop_connection(self):
        stream, self.stream = self.stream, None
        if stream is not None:
            with suppress(Exception):
                stream.close()

    def reconnect_deadline_expired(self):
        return (
            self.reconnect_started is not None
            and time.monotonic() - self.reconnect_started >= self.reconnect_window
        )

    def idle_action(self, connection_started):
        now = time.monotonic()
        if self.auth_deadline and not self.authenticated and now > self.auth_deadline:
            raise DuelError("account_login_timeout")
        if self.terminal_time is not None:
            return
        if self.reconnect_deadline_expired():
            raise ReconnectRequired("tournament reconnect handshake timed out")
        idle = now - (self.last_packet_at or connection_started)
        if self.lifecycle_state in ("connecting", "joining", "reconnecting"):
            if now - connection_started >= self.connect_window:
                raise ReconnectRequired("room handshake timed out")
            return
        if self.lifecycle_state == "waiting":
            # Tournament Match rooms can legitimately wait while the other
            # participant or its model is still loading.
            if self.match_mode:
                return
            if idle >= self.timeout:
                raise socket.timeout("single-duel lobby timed out")
            return
        if self.lifecycle_state in ("saving", "siding"):
            if idle >= self.transition_idle:
                raise ReconnectRequired("next tournament game did not start")
            return
        if self.lifecycle_state in ("selecting_turn", "dueling") and idle >= self.timeout:
            raise ReconnectRequired("tournament duel transport became idle")

    def _handle(self, op, payload):
        if op == deck_sync.OPCODE:
            if not self.deck_sync_protocol and self.deck_source == "local":
                return
            if not self.deck_sync_protocol or self.pending_deck is not None or (
                    self.started.is_set() and not self.terminal_seen
                    and not self.resume_current_game):
                raise DuelError("deck_sync_unexpected_manifest")
            self.pending_deck = deck_sync.decode(payload, self.sync_nonce, len(self.games) + 1)
            return
        if op == P.STOC.CHAT:
            if self.auth_mode == "duelchronicle" and not self.authenticated:
                text = P.decode_name(payload[2:])
                channel = int.from_bytes(payload[:2], "little")
                if self.auth_sent and channel >= 11 and text.startswith("[Server]:"):
                    if "\u767b\u5f55\u6210\u529f" in text:
                        self.authenticated = True
                        self._submit_deck()
                    elif any(word in text for word in (
                            "\u5bc6\u7801\u9519\u8bef", "\u767b\u5f55\u5931\u8d25",
                            "\u5c1a\u672a\u6ce8\u518c", "\u672a\u6ce8\u518c")):
                        raise DuelError("account_login_failed")
            return
        if op == P.STOC.REPLAY:
            digest = hashlib.sha256(payload).hexdigest()
            if digest in self.replay_hashes:
                return
            try:
                self.save_replay(payload)
            except (ValueError, OSError, lzma.LZMAError) as error:
                # Recording is secondary to play. RecordingStream has already
                # journaled these original bytes durably, including rejected data.
                message = type(error).__name__ + ": " + str(error)[:250]
                if message not in self.replay_errors and len(self.replay_errors) < 10:
                    self.replay_errors.append(message)
                return
            self.replay_hashes.add(digest)
            self.replay_received = True
            return
        if op == P.STOC.TYPE_CHANGE and payload:
            self.host_player = bool(payload[0] & 0x10)
        if op == P.STOC.SELECT_TP:
            self.change_state("selecting_turn", preferred_first=self.go_first)
        if op == P.STOC.JOIN_GAME:
            self.change_state("joining")
            self.host_info = P.HostInfo.unpack(payload)
            self.match_mode = self.host_info.mode == 1
            self.ever_joined = True
            self.reconnect_started = None
            self.reconnect_attempts = 0
            if self.host_info.mode != 0 and not self.match_mode:
                raise DuelError(f"room mode {self.host_info.mode} is not a single duel")
            if self.authenticated:
                self._submit_deck()
            return
        if op == P.STOC.TYPE_CHANGE:
            self.lobby_pos = payload[0] & 0xF
            if self.lobby_pos == P.NETPLAYER_TYPE_OBSERVER:
                self.stream.send(P.CTOS.HS_TODUELIST)
                return
            if not self.authenticated:
                if not self.auth_sent:
                    self.change_state("authenticating")
                    self.stream.send(P.CTOS.CHAT, (
                        "/dl " + self.account_password + "\0").encode("utf-16-le"))
                    self.auth_sent = True
                    self.auth_deadline = time.monotonic() + 15
            elif self.auth_mode == "none":
                self.stream.send(P.CTOS.HS_READY)
                self.joined.set()
                self.change_state("waiting")
            return
        if op in (P.STOC.CHANGE_SIDE, P.STOC.WAITING_SIDE) and self.match_mode:
            if not self.terminal_seen:
                raise DuelError("Match side transition without a game result")
            self.change_state("siding", games=len(self.games))
            if op == P.STOC.CHANGE_SIDE and self.auth_mode == "none":
                # Fixed-deck tournament rooms keep the same submitted deck.
                self.stream.send(
                    P.CTOS.UPDATE_DECK,
                    P.update_deck(self.main + self.extra, self.side),
                )
            return
        if op == P.STOC.ERROR_MSG:
            if len(payload) < 8:
                raise DuelError("Truncated server error")
            category, code = payload[0], int.from_bytes(payload[4:8], "little")
            names = {1: "join_rejected", 2: "deck_rejected", 3: "side_deck_rejected",
                     4: "protocol_version_mismatch"}
            raise DuelError(f"{names.get(category, 'server_error')}: {code}")
        if op == P.STOC.DUEL_END:
            # Some hosts deliver REPLAY immediately after DUEL_END.
            if self.terminal_seen:
                if self.match_mode:
                    self.match_done = True
                    self.terminal_time = time.monotonic()
                return
            raise DuelError("Server ended duel without an authoritative result")
        super()._handle(op, payload)
        if op == P.STOC.TYPE_CHANGE and self.lobby_pos in (0, 1):
            self.change_state("waiting")
        if op == P.STOC.HS_PLAYER_CHANGE and payload:
            if payload[0] >> 4 == self.lobby_pos:
                self.own_ready = payload[0] & 0xF == P.PLAYERCHANGE.READY
            if self.host_player and self.opponent_ready.is_set() and self.authenticated and (
                    self.auth_mode == "none" or self.own_ready):
                self.stream.send(P.CTOS.HS_START)

    def _submit_deck(self):
        if self.submitted:
            return
        self.stream.send(P.CTOS.UPDATE_DECK, P.update_deck(self.main + self.extra, self.side))
        self.submitted = True
        self.joined.set()
        self.change_state("waiting")

    def _game_msg(self, payload):
        if not payload:
            raise DuelError("Empty game message")
        msg, body = payload[0], payload[1:]
        if msg == C.MSG_START:
            resuming = self.resume_current_game
            if resuming:
                self.selector = None
            elif self.games:
                if not self.match_mode or not self.terminal_seen or self.match_done:
                    raise DuelError("Unexpected additional duel")
                if len(self.games) >= MAX_MATCH_GAMES:
                    raise DuelError("Match game limit exceeded")
                self.result = DuelResult()
                self.terminal_seen = self.replay_received = False
                self.selector = None
                self.terminal_time = None
            if resuming or self.games:
                if self.deck_source == "server_random" or self.deck_sync_protocol:
                    from .server_deck import ServerDeckBoard
                    self.board = ServerDeckBoard()
                    known_codes = []
                else:
                    self.board = ShadowBoard()
                    known_codes = list(dict.fromkeys(
                        self.main + self.extra + self.side
                    ))
                self.ctx = SelectContext(
                    max_options=self.max_options,
                    rng=self.rng,
                    card_pool=self.card_pool,
                )
                self.ctx.known_codes = known_codes
            self.deck_sync = {"state": "pending" if self.deck_source == "server_random" or
                              self.deck_sync_protocol else "local"}
            if self.deck_sync_protocol and self.pending_deck is None:
                raise DuelError("deck_sync_missing_manifest")
        if self.version == 0x133E:
            body = legacy_message(msg, body)
        if (
            msg == C.MSG_WIN
            and self.terminal_seen
            and self.match_mode
            and body[:2] == bytes((2, 0x11))
        ):
            # Evolution may report a shutdown failure after the authoritative
            # result has already been delivered. It is not a second game.
            return
        self._track(msg, body)
        self.board.apply(msg, body)
        if msg == C.MSG_START and self.pending_deck is not None:
            document, deck = self.pending_deck
            self.pending_deck = None
            self.board.assign(deck)
            self.ctx.known_codes = list(dict.fromkeys(deck["main"] + deck["extra"] + deck["side"]))
            self.deck_sync = {"state": "synchronized", "source": "relay-v1",
                              "game": document["game"], "sha256": document["sha256"],
                              "counts": {k: len(v) for k, v in deck.items()}}
            self.assigned_deck(document, deck)
        if msg == C.MSG_START and self.resume_current_game:
            self.resume_current_game = False
        self.policy.observe_game_message(msg, body)
        if msg == C.MSG_START:
            counts = struct.unpack_from("<4H", body, 10)
            self.initial_deck_counts = [list(counts[:2]), list(counts[2:])]
            self.change_state("dueling", model_seat=self.result.our_player, deck_sync=self.deck_sync)
        if msg == C.MSG_WIN:
            if self.terminal_seen:
                raise DuelError("Duplicate game result")
            self.terminal_seen = True
            self.terminal_time = None if self.match_mode else time.monotonic()
            self._duel_over = False
            self.result.lp = tuple(self._lp)
            game = {**self.result.as_dict(), "game": len(self.games) + 1,
                    "initial_deck_counts": self.initial_deck_counts, "deck_sync": dict(self.deck_sync)}
            self.games.append(game)
            self.game_finished(game)
            self.change_state("saving", games=len(self.games))
        elif msg == C.MSG_MATCH_KILL:
            self.match_kill = True
        elif msg in SELECT_MESSAGES:
            self._answer(msg, body)

    def _answer(self, msg, body):
        if self.require_deck_info and (self.deck_source == "server_random" or self.deck_sync_protocol):
            try:
                self.board.require_complete()
            except ValueError:
                self.deck_sync = {**self.deck_sync, "state": "unavailable"}
                self.change_state("dueling", deck_sync=self.deck_sync)
                raise
            if self.deck_sync["state"] != "synchronized":
                self.deck_sync = {"state": "synchronized", "source": "server_refresh",
                                  "game": len(self.games) + 1}
                self.change_state("dueling", deck_sync=self.deck_sync)
        result = parse_prompt(msg, body, self.ctx)
        if result.player != self.result.our_player:
            raise DuelError("Received another player's private prompt")
        if result.cards:
            problems = self.board.cross_check(result.cards)
            if problems:
                raise DuelError("Board/prompt disagreement: " + str(problems[0])[:200])
        if result.auto_response is not None:
            self.result.auto_responses += 1
            self._send_response(result.auto_response)
            return
        selector = self.selector = result.selector
        for _ in range(256):
            actions = selector.options()
            if not 1 <= len(actions) <= self.max_options:
                raise DuelError("Invalid legal-action count")
            state = self._decision_state(selector, actions)
            if len(actions) == 1:
                index, forced = 0, True
                self.result.forced_actions += 1
            else:
                index, forced = self.policy.choose(state), False
                if type(index) is not int or not 0 <= index < len(actions):
                    raise DuelError("Invalid model action")
                self.result.decisions += 1
                if self.result.decisions > 1000:
                    raise DuelError("Decision limit reached")
            action = actions[index]
            response = selector.choose(index)
            self.policy.record(state, action, selector, index, forced)
            if response is not None:
                self._send_response(response)
                return
        raise DuelError("Selection did not finish")

    def run(self):
        start = time.monotonic()
        connection_started = start
        connect_deadline = start + self.connect_window
        try:
            while not self.cancelled:
                try:
                    reconnecting = self.ever_joined or self.reconnect_started is not None
                    self.connect(reconnecting=reconnecting)
                    connection_started = time.monotonic()
                    while not self.cancelled:
                        replays_complete = (len(self.replay_hashes) >= len(self.games)
                                            if self.match_mode else self.replay_received)
                        if self.terminal_time is not None and (
                                replays_complete or time.monotonic() - self.terminal_time > 8):
                            return self.result
                        try:
                            op, payload = self.stream.recv(
                                min(1, self.poll_interval)
                                if self.terminal_time is not None or not self.authenticated
                                else self.poll_interval
                            )
                        except socket.timeout:
                            self.idle_action(connection_started)
                            continue
                        self.last_packet_at = time.monotonic()
                        self._handle(op, payload)
                except (ConnectionError, OSError) as error:
                    if self.settled:
                        break
                    prior = self.lifecycle_state
                    if prior == "dueling":
                        self.resume_current_game = True
                    self.drop_connection()
                    if not self.ever_joined:
                        if time.monotonic() >= connect_deadline:
                            raise TimeoutError("connection_retry_timeout") from error
                    elif not self.match_mode:
                        raise
                    else:
                        if self.reconnect_started is None:
                            self.reconnect_started = time.monotonic()
                        if self.reconnect_deadline_expired():
                            raise TimeoutError("tournament_reconnect_timeout") from error
                    self.reconnect_attempts += 1
                    delay = self.reconnect_backoff[
                        min(self.reconnect_attempts - 1, len(self.reconnect_backoff) - 1)
                    ]
                    self.change_state(
                        "reconnecting",
                        attempt=self.reconnect_attempts,
                        retry_in=delay,
                        previous_state=prior,
                    )
                    time.sleep(delay)
        except Exception as error:
            message = str(error)
            if self.account_password:
                message = message.replace(self.account_password, "[REDACTED]")
            if self.password:
                message = message.replace(self.password, "[REDACTED]")
            self.result.error = type(error).__name__ + ": " + message[:300]
            if not isinstance(error, (ConnectionError, OSError)):
                self.surrender()
        finally:
            if self.cancelled and not self.settled:
                self.result.error = "stopped_by_owner"
            self.result.seconds = time.monotonic() - start
            self.result.lp = tuple(self._lp)
            self.close()
        return self.result

    @property
    def settled(self):
        return self.terminal_seen and (not self.match_mode or self.match_done)

    def outcome(self):
        if not self.settled or self.result.our_player not in (0, 1) or self.result.winner not in (0, 1, 2):
            return "incomplete"
        if not self.match_mode or self.match_kill:
            return "draw" if self.result.winner == 2 else "win" if self.result.won else "loss"
        wins = sum(game["winner"] == game["our_player"] for game in self.games)
        losses = sum(game["winner"] in (0, 1) and game["winner"] != game["our_player"] for game in self.games)
        return "win" if wins > losses else "loss" if losses > wins else "draw"
