try:
    import TC300_COMMAND_LIB
    from TC300_COMMAND_LIB import TC300ListDevices, TC300Open, TC300SetTargetTemperature, TC300GetActualTemperature, \
        TC300Close
except (ModuleNotFoundError, ImportError):
    # this is installed with the TC300 sdk from thorlabs:
    # https://www.thorlabs.com/newgrouppage9.cfm?objectgroup_id=14852
    print('couldn\'t import TC300_COMMAND_LIB')
import time


class TC300(object):
    # note: added to path and to pythonpath this dir: C:\Program Files (x86)\Thorlabs\TC300\Sample\Thorlabs_TC300_PythonSDK
    # note also that in TC300_COMMAND_LIB.py you need th change the LoadLibrary to load the 64x instead of the win32

    def __init__(self):
        self.devs = TC300ListDevices()
        self.TC300 = self.devs[0]
        self.serialNumber = self.TC300[0]
        self.hdl = TC300Open(self.serialNumber, 115200, 3)

    def set_temperature(self, temperature):
        assert temperature < 75
        res = TC300SetTargetTemperature(self.hdl, 1, temperature)
        assert res == 0

    def get_temperature(self):
        ActualTemperature = [0]
        result = TC300GetActualTemperature(self.hdl, 1, ActualTemperature)
        return ActualTemperature[0]

    def wait_for_temp(self, temperature):
        while self.get_temperature() > temperature + 0.25 or self.get_temperature() < temperature - 0.25:
            time.sleep(0.3)

    def close(self):
        # TODO: close better such that after closing you can open it without closing ipython
        TC300Close(self.hdl)
