import serial


class ElliptecMotor(object):
    _PULS_PER_MM = 398  # Got this magic number from the GUI under "details"

    def __init__(self, port_no=7):
        self.port_no = port_no
        self.ser = serial.Serial(f'COM{port_no}')
        self.motor_no = 0  # "Address string"

        self.serial_no = None
        self.jog_step_size = None
        self.home_offset = None
        self.current_position = None

    def _scan_for_devices(self):
        ans = self.send_command('in')

        self.serial_no = ans

    def jog_forward(self):
        _, cmdid, val = self.send_command('fw')
        assert cmdid == 'GS'
        assert val == '00'

    def jog_backwards(self):
        _, cmdid, val = self.send_command('bw')
        assert cmdid == 'GS'
        assert val == '00'
        # val of 02 means mechanical timeout

    def set_jog_step(self, step_size):
        step_size = self._mm_to_pulse_8byte_hex_str(step_size)
        _, cmdid, val = self.send_command('sj', step_size)
        assert cmdid == 'GS'
        assert val == '00'
        print('OK')

    def get_jog_step(self):
        _, cmdid, val = self.send_command('gj')
        assert cmdid == 'GJ'
        val = self._pulse_hex_str_to_mm(val)
        self.jog_step_size = val
        return val

    def get_home_offset(self):
        _, cmdid, val = self.send_command('go')
        assert(cmdid == 'HO')
        self.home_offset = self._pulse_hex_str_to_mm(val)
        return self.home_offset

    def set_home_offset(self, offset):
        offset = self._mm_to_pulse_8byte_hex_str(offset)
        _, cmdid, val = self.send_command('so', offset)
        assert cmdid in ('PO', 'GS')
        assert val == '00'
        print('OK')

    def move_absolute(self, degs):
        assert 0 <= degs <= 360, 'Can\'t move absolute to more than 360 degrees'
        degs = self._mm_to_pulse_8byte_hex_str(degs)
        _, cmdid, val =self.send_command('ma', degs)
        assert cmdid in ('PO', 'GS')
        if cmdid == 'GS':
            assert val == '00'
        # print('OK')

    def move_relative(self, degs):
        degs = self._mm_to_pulse_8byte_hex_str(degs)
        _, cmdid, val = self.send_command('mr', degs)
        assert cmdid in ('PO', 'GS')
        self.current_position = self._pulse_hex_str_to_mm(val)
        # print('OK')

    def get_position(self):
        _, cmdid, val = self.send_command('gp')
        assert cmdid in ('PO', 'GS')
        self.current_position = self._pulse_hex_str_to_mm(val)
        return self.current_position

    def move_home(self):
        _, cmdid, val = self.send_command('ho', '1')
        assert cmdid in ('PO', 'GS')
        self.current_position = self._pulse_hex_str_to_mm(val)
        print('OK')

    def send_command(self, command_type, command_val=''):
        self.ser.write(f'{self.motor_no}{command_type}{command_val}'.encode())

        ans = self.ser.read_until().decode()
        address = int(ans[0])
        response_type = ans[1:3]
        data = ans[3:].strip()

        return address, response_type, data

    def close(self):
        self.ser.close()

    @classmethod
    def _mm_to_pulse_8byte_hex_str(cls, pos):
        val = round(pos * cls._PULS_PER_MM)
        hex_str = cls._int2dword(int(val))
        return hex_str

    @classmethod
    def _pulse_hex_str_to_mm(cls, hex_str):
        raw_pos = int(hex_str, 16)
        if raw_pos > 0xffffffff // 2:
            raw_pos = -(0xffffffff - raw_pos)
        pos = raw_pos / cls._PULS_PER_MM  # [mm]

        return pos

    @classmethod
    def _int2word(cls, val: int):
        assert(0 <= val <= 65535)
        # Due to the protocol, X must be in upper case
        hex_str = format(val, '04X')  # Hex length 4
        return hex_str

    @classmethod
    def _int2dword(cls, val: int):
        assert 0 <= abs(val) <= 0xffffffff // 2

        if val < 0:
            val = 0xffffffff + val

        # Due to the protocol, X must be in upper case
        hex_str = format(val, '08X')  # Hex length 8
        return hex_str
