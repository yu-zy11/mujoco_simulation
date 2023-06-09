"""LCM type definitions
This file automatically generated by lcm.
DO NOT MODIFY BY HAND!!!!
"""

try:
    import cStringIO.StringIO as BytesIO
except ImportError:
    from io import BytesIO
import struct

class MujocoState(object):
    __slots__ = ["q_abad", "q_hip", "q_knee", "qd_abad", "qd_hip", "qd_knee", "tau_abad", "tau_hip", "tau_knee", "imu_acc", "imu_omega", "imu_quat", "body_position", "body_velocity"]

    __typenames__ = ["float", "float", "float", "float", "float", "float", "float", "float", "float", "float", "float", "float", "float", "float"]

    __dimensions__ = [[4], [4], [4], [4], [4], [4], [4], [4], [4], [3], [3], [4], [3], [3]]

    def __init__(self):
        self.q_abad = [ 0.0 for dim0 in range(4) ]
        self.q_hip = [ 0.0 for dim0 in range(4) ]
        self.q_knee = [ 0.0 for dim0 in range(4) ]
        self.qd_abad = [ 0.0 for dim0 in range(4) ]
        self.qd_hip = [ 0.0 for dim0 in range(4) ]
        self.qd_knee = [ 0.0 for dim0 in range(4) ]
        self.tau_abad = [ 0.0 for dim0 in range(4) ]
        self.tau_hip = [ 0.0 for dim0 in range(4) ]
        self.tau_knee = [ 0.0 for dim0 in range(4) ]
        self.imu_acc = [ 0.0 for dim0 in range(3) ]
        self.imu_omega = [ 0.0 for dim0 in range(3) ]
        self.imu_quat = [ 0.0 for dim0 in range(4) ]
        self.body_position = [ 0.0 for dim0 in range(3) ]
        self.body_velocity = [ 0.0 for dim0 in range(3) ]

    def encode(self):
        buf = BytesIO()
        buf.write(MujocoState._get_packed_fingerprint())
        self._encode_one(buf)
        return buf.getvalue()

    def _encode_one(self, buf):
        buf.write(struct.pack('>4f', *self.q_abad[:4]))
        buf.write(struct.pack('>4f', *self.q_hip[:4]))
        buf.write(struct.pack('>4f', *self.q_knee[:4]))
        buf.write(struct.pack('>4f', *self.qd_abad[:4]))
        buf.write(struct.pack('>4f', *self.qd_hip[:4]))
        buf.write(struct.pack('>4f', *self.qd_knee[:4]))
        buf.write(struct.pack('>4f', *self.tau_abad[:4]))
        buf.write(struct.pack('>4f', *self.tau_hip[:4]))
        buf.write(struct.pack('>4f', *self.tau_knee[:4]))
        buf.write(struct.pack('>3f', *self.imu_acc[:3]))
        buf.write(struct.pack('>3f', *self.imu_omega[:3]))
        buf.write(struct.pack('>4f', *self.imu_quat[:4]))
        buf.write(struct.pack('>3f', *self.body_position[:3]))
        buf.write(struct.pack('>3f', *self.body_velocity[:3]))

    def decode(data):
        if hasattr(data, 'read'):
            buf = data
        else:
            buf = BytesIO(data)
        if buf.read(8) != MujocoState._get_packed_fingerprint():
            raise ValueError("Decode error")
        return MujocoState._decode_one(buf)
    decode = staticmethod(decode)

    def _decode_one(buf):
        self = MujocoState()
        self.q_abad = struct.unpack('>4f', buf.read(16))
        self.q_hip = struct.unpack('>4f', buf.read(16))
        self.q_knee = struct.unpack('>4f', buf.read(16))
        self.qd_abad = struct.unpack('>4f', buf.read(16))
        self.qd_hip = struct.unpack('>4f', buf.read(16))
        self.qd_knee = struct.unpack('>4f', buf.read(16))
        self.tau_abad = struct.unpack('>4f', buf.read(16))
        self.tau_hip = struct.unpack('>4f', buf.read(16))
        self.tau_knee = struct.unpack('>4f', buf.read(16))
        self.imu_acc = struct.unpack('>3f', buf.read(12))
        self.imu_omega = struct.unpack('>3f', buf.read(12))
        self.imu_quat = struct.unpack('>4f', buf.read(16))
        self.body_position = struct.unpack('>3f', buf.read(12))
        self.body_velocity = struct.unpack('>3f', buf.read(12))
        return self
    _decode_one = staticmethod(_decode_one)

    _hash = None
    def _get_hash_recursive(parents):
        if MujocoState in parents: return 0
        tmphash = (0xb9325c5b17cea306) & 0xffffffffffffffff
        tmphash  = (((tmphash<<1)&0xffffffffffffffff) + (tmphash>>63)) & 0xffffffffffffffff
        return tmphash
    _get_hash_recursive = staticmethod(_get_hash_recursive)
    _packed_fingerprint = None

    def _get_packed_fingerprint():
        if MujocoState._packed_fingerprint is None:
            MujocoState._packed_fingerprint = struct.pack(">Q", MujocoState._get_hash_recursive([]))
        return MujocoState._packed_fingerprint
    _get_packed_fingerprint = staticmethod(_get_packed_fingerprint)

