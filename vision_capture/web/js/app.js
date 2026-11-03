function pollHardwareStatus() {
    fetch('/api/hardware_status')
    .then(res => res.json())
    .then(data => {
        const envStatus = document.getElementById('env-status');
        const envIcon = document.getElementById('env-icon');
        
        if (data.jetson_connected) {
            // UPDATED: Clinically focused messaging
            envStatus.innerHTML = 'Ready for PLR Evaluation<br><span style="color:#4ade80; font-size:0.9rem;">Automated Diabetic Retinopathy Screening</span>';
            envIcon.style.color = '#4ade80';
        } else {
            envStatus.innerHTML = 'Windows / x86 Host<br><span style="color:#facc15; font-size:0.9rem;">No diagnostic hardware linked</span>';
            envIcon.style.color = '#facc15';
        }

        const camStatus = document.getElementById('cam-status');
        const camIcon = document.getElementById('cam-icon');
        if (data.cameras_detected === 2) {
            camStatus.innerHTML = 'Dual NIR Online<br><span style="color:#4ade80; font-size:0.9rem;">2 of 2 sensors active</span>';
            camIcon.style.color = '#4ade80';
        } else if (data.cameras_detected === 1) {
            camStatus.innerHTML = 'Single Camera<br><span style="color:#facc15; font-size:0.9rem;">1 of 2 sensors active</span>';
            camIcon.style.color = '#facc15';
        } else {
            camStatus.innerHTML = 'Optics Offline<br><span style="color:#ff4c4c; font-size:0.9rem;">0 sensors detected</span>';
            camIcon.style.color = '#ff4c4c';
        }
    })
    .catch(err => console.error("Hardware probe failed:", err));
}

async function pollSystemResources() {
    try {
        const response = await fetch("/api/sys_resources");
        const data = await response.json();
        
        if (data.ok) {
            const fanPercent = data.fan_pwm > 0 ? Math.round((data.fan_pwm / 255) * 100) : 0;
            document.getElementById("sys-fan").innerText = `${fanPercent}%`;
            document.getElementById("sys-ram").innerText = `${data.ram_percent}%`;
            document.getElementById("sys-storage").innerText = `${data.disk_percent}%`;
            
            if (data.temp_c !== undefined) {
                let tempEl = document.getElementById("sys-temp");
                let tempVal = parseFloat(data.temp_c).toFixed(1);
                tempEl.innerText = `${tempVal}°C`;
                
                if (tempVal >= 70) tempEl.style.color = "#ef4444";
                else tempEl.style.color = "#4ade80";
            }
        }
    } catch (err) { console.error("System resource fetch failed:", err); }
}

function launchTool(toolId) {
    if (toolId === 'recorder') {
        window.open('/recorder', '_blank');
    }
}

let batteryWarningShown = false;

async function pollBattery() {
    try {
        const response = await fetch("/api/battery");
        const data = await response.json();
        if (data.ok) {
            document.getElementById("batt-text").innerText = `${Math.round(data.percentage)}%`;
            document.getElementById("batt-fill").style.width = `${data.percentage}%`;
            const voltEl = document.getElementById("batt-volt");
            if (voltEl) voltEl.innerText = `${data.voltage}V`;
            const container = document.getElementById("batt-container");
            const fill = document.getElementById("batt-fill");
            
            if (data.charging) {
                container.classList.add("batt-charging");
                fill.style.background = "#4ade80";
                batteryWarningShown = false;
            } else {
                container.classList.remove("batt-charging");
                fill.style.background = data.percentage < 20 ? "#ef4444" : "#cbd5e1";
                
                if (data.percentage <= 45 && !batteryWarningShown) {
                    showBatteryWarning();
                    batteryWarningShown = true;
                }
            }
        }
    } catch (err) { console.error("Battery fetch failed:", err); }
}

function showBatteryWarning() {
    const notif = document.createElement("div");
    notif.style.position = "fixed";
    notif.style.top = "20px";
    notif.style.left = "50%";
    notif.style.transform = "translateX(-50%)";
    notif.style.background = "rgba(239, 68, 68, 0.95)";
    notif.style.color = "white";
    notif.style.padding = "15px 30px";
    notif.style.borderRadius = "8px";
    notif.style.boxShadow = "0 4px 20px rgba(239, 68, 68, 0.6)";
    notif.style.fontWeight = "bold";
    notif.style.zIndex = "9999";
    notif.style.textAlign = "center";
    notif.style.border = "1px solid #fca5a5";
    notif.innerHTML = "⚠️ LOW BATTERY DETECTED!<br><span style='font-size: 0.8rem; font-weight: normal;'>Please connect the charger to prevent sudden shutdown.</span>";
    
    document.body.appendChild(notif);
    
    setTimeout(() => {
        if(notif.parentNode) notif.parentNode.removeChild(notif);
    }, 6000);
}

document.addEventListener('DOMContentLoaded', () => {
    pollHardwareStatus();
    setInterval(pollHardwareStatus, 5000);
    
    pollSystemResources();
    setInterval(pollSystemResources, 5000);

    pollBattery();
    setInterval(pollBattery, 15000);

    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        setTimeout(() => { overlay.classList.add('is-hidden'); }, 300);
    }
});

function refreshDashboard() {
    const overlay = document.getElementById('loading-overlay');
    const loadingText = document.getElementById('loading-text');

    if (loadingText) loadingText.textContent = 'Refreshing TANGLAWAN AI Tools...';
    if (overlay) overlay.classList.remove('is-hidden');

    setTimeout(() => { location.reload(); }, 250);
}

function shutdownSystem() {
    if (confirm("WARNING: Are you sure you want to safely power off the TANGLAWAN hardware?")) {
        let btn = document.getElementById('btn-sys-shutdown');
        btn.innerHTML = "⌛";
        btn.disabled = true;
        
        fetch('/api/system/shutdown', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                if(data.ok) showShutdownScreen();
                else {
                    alert("Shutdown command failed: " + data.error);
                    btn.innerHTML = "⏻";
                    btn.disabled = false;
                }
            })
            .catch(err => { showShutdownScreen(); });
    }
}

function showShutdownScreen() {
    document.body.innerHTML = `
        <div style="display: flex; justify-content: center; align-items: center; height: 100vh; background-color: #010409; color: #e2e8f0; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; text-align: center; flex-direction: column;">
            <h1 style="color: #4ade80; font-size: 3rem; margin-bottom: 10px; letter-spacing: 2px;">HARDWARE HALTED</h1>
            <p style="color: #94a3b8; font-size: 1.2rem;">The OS has safely shut down.<br>Please wait 5 seconds, then disconnect the power supply.</p>
        </div>
    `;
}