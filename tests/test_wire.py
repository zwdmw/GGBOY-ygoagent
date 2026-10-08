import socket
import unittest

from ygo_sky.duel.network.client import RecordingStream
from ygo_sky.duel.network.vendor.netduel import protocol as P


class WireTests(unittest.TestCase):
    def test_sensitive_join_is_redacted_but_wire_is_preserved(self):
        left, right = socket.socketpair()
        rows = []
        try:
            stream = RecordingStream(left, lambda *args, **fields: rows.append((args, fields)))
            payload = P.join_game(0x1362, password="fixture-room")
            stream.send(P.CTOS.JOIN_GAME, payload)
            opcode, actual = P.PacketStream(right).recv(timeout=1)
            self.assertEqual((opcode, actual), (P.CTOS.JOIN_GAME, payload))
            self.assertEqual(rows[0][0][2], b"")
            self.assertTrue(rows[0][1]["redacted"])
        finally:
            left.close()
            right.close()

    def test_fragmented_packet_is_reassembled(self):
        left, right = socket.socketpair()
        try:
            raw = P.pack_packet(P.STOC.GAME_MSG, b"fixture")
            left.sendall(raw[:1])
            left.sendall(raw[1:3])
            left.sendall(raw[3:])
            self.assertEqual(P.PacketStream(right).recv(timeout=1), (P.STOC.GAME_MSG, b"fixture"))
        finally:
            left.close()
            right.close()
