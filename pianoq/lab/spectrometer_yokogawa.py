import vxi11

class YokogawaSpectrometer(object):
    def __init__(self):
        self.instr = vxi11.Instrument("192.168.1.110")
        assert self.instr.ask('*IDN?') == 'YOKOGAWA,AQ6374,91V714535,01.03'

    def sweep(self):
        self.instr.write(":sens:wav:cent 1550nm") # sweep center wl
        self.instr.write(":sens:wav:span 10nm") # sweep span
        self.instr.write(":sens:sens mid") # sens mode = MID
        self.instr.write(":sens:sweep:points:auto on") # Sampling
        pass


def foo():
    instr =  vxi11.Instrument("192.168.1.110")
    print(instr.ask("*IDN?")) # sanity
    instr.write("CFORM1")
    instr.write(":sens:wav:cent 780nm") # sweep center wl
    instr.write(":sens:wav:span 50nm") # sweep span
    instr.write(":sens:sens mid") # sens mode = MID
    instr.write(":sens:sweep:points:auto on") # Sampling
    instr.write(":init:smode single")
    instr.write("*CLS")
    instr.write(":init")
    # 0 means not done yet, 1 means done
    while instr.ask(":stat:oper:even?") == 0:
        time.sleep(0.01)

    wl = instr.ask(":TRACE:X? TRA")
    amp = instr.ask(":trac:y:pden? tra,0.1nm")

    wl = [float(x) for x in wl.split(',')]
    amp = [float(x) for x in amp.split(',')]
    mplot(wl, amp)
