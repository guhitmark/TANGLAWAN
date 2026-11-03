import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "UPS_Power_Module_B"))

try:
    from INA219 import INA219
    _ina = INA219(addr=0x42)
    HAS_INA = True
except Exception:
    HAS_INA = False

def read_battery_status():
    if not HAS_INA:
        return {"ok": False, "percentage": 0, "percent": "N/A", "voltage": 0.0, "charging": False}
    try:
        bus_voltage = _ina.getBusVoltage_V()
        current_mA  = _ina.getCurrent_mA()

        # Waveshare UPS Module B formula (2S LiPo: 6.0V–8.4V)
        p = (bus_voltage - 6.0) / 2.4 * 100.0
        p = max(0.0, min(100.0, p))

        charging = current_mA > 0.0

        return {
            "ok":         True,
            "percentage": round(p, 1),
            "percent":    f"{round(p, 1)}%",
            "voltage":    round(bus_voltage, 3),
            "charging":   charging,
            "current_mA": round(current_mA, 2),
        }
    except Exception as e:
        return {"ok": False, "percentage": 0, "percent": "ERR", "voltage": 0.0, "charging": False}