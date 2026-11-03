const API = "/api/recorder";

const defaultCrop = {
    L: [[10, 10], [390, 10], [390, 390], [10, 390]],
    R: [[10, 10], [390, 10], [390, 390], [10, 390]],
};
let cropPoints = JSON.parse(JSON.stringify(defaultCrop));

let dragging = { L: null, R: null };
let dragStart = { L: null, R: null };
let activeLabEye = null;

let sessionHistory = { L: [], R: [] }; 

let redChartInstance = null;
let blueChartInstance = null;

function getCropMode() { return document.querySelector('input[name="cropMode"]:checked').value; }

function drawCanvas(eye) {
    const chk = document.getElementById(eye === 'L' ? 'enableLeft' : 'enableRight');
    if (chk && !chk.checked) return; 

    const canvas = document.getElementById(`crop-canvas-${eye}`);
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    const pts = cropPoints[eye];
    ctx.strokeStyle = "#facc15"; ctx.lineWidth = 4; ctx.beginPath();
    pts.forEach((p, i) => (i === 0 ? ctx.moveTo(p[0], p[1]) : ctx.lineTo(p[0], p[1])));
    ctx.closePath(); ctx.stroke();
    pts.forEach((p) => { ctx.fillStyle = "#facc15"; ctx.beginPath(); ctx.arc(p[0], p[1], 12, 0, Math.PI * 2); ctx.fill(); });
}

function applyDisabledOverlay(eye, isEnabled) {
    ['raw', 'crop'].forEach(type => {
        const img = document.getElementById(`${type}-${eye}`);
        if (!img) return;
        
        const wrapperId = `disabled-overlay-${type}-${eye}`;
        let overlay = document.getElementById(wrapperId);
        
        if (!isEnabled) {
            img.style.filter = "grayscale(100%) opacity(20%)";
            if (!overlay) {
                const parent = img.parentElement;
                if (parent) {
                    parent.style.position = 'relative';
                    overlay = document.createElement('div');
                    overlay.id = wrapperId;
                    overlay.innerText = "DISABLED";
                    overlay.style.position = "absolute";
                    overlay.style.top = "50%";
                    overlay.style.left = "50%";
                    overlay.style.transform = "translate(-50%, -50%)";
                    overlay.style.color = "#ef4444"; 
                    overlay.style.fontWeight = "bold";
                    overlay.style.fontSize = "1.5rem";
                    overlay.style.letterSpacing = "2px";
                    overlay.style.zIndex = "20";
                    overlay.style.pointerEvents = "none";
                    parent.appendChild(overlay);
                }
            }
            if (overlay) overlay.style.display = "block";
        } else {
            img.style.filter = "none";
            if (overlay) overlay.style.display = "none";
        }
    });
    
    const canvas = document.getElementById(`crop-canvas-${eye}`);
    if (canvas) {
        const ctx = canvas.getContext("2d");
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        canvas.style.pointerEvents = isEnabled ? "auto" : "none";
        if (isEnabled) drawCanvas(eye);
    }
}

function setupCanvas(eye) {
    const canvas = document.getElementById(`crop-canvas-${eye}`);
    if (!canvas) return;
    
    canvas.width = 400;
    canvas.height = 400;

    function toCanvasCoords(evt) {
        const rect = canvas.getBoundingClientRect();
        return [(evt.clientX - rect.left) * (canvas.width / rect.width), (evt.clientY - rect.top) * (canvas.height / rect.height)];
    }

    canvas.addEventListener("mousedown", (evt) => {
        const [x, y] = toCanvasCoords(evt);
        for (let i = 0; i < cropPoints[eye].length; i++) {
            if (Math.hypot(x - cropPoints[eye][i][0], y - cropPoints[eye][i][1]) < 30) {
                dragging[eye] = i; return;
            }
        }
        
        let pts = cropPoints[eye];
        let minX = Math.min(pts[0][0], pts[1][0], pts[2][0], pts[3][0]);
        let maxX = Math.max(pts[0][0], pts[1][0], pts[2][0], pts[3][0]);
        let minY = Math.min(pts[0][1], pts[1][1], pts[2][1], pts[3][1]);
        let Math_maxY = Math.max(pts[0][1], pts[1][1], pts[2][1], pts[3][1]);
        
        if (x >= minX && x <= maxX && y >= minY && y <= Math_maxY) {
            dragging[eye] = 'all';
            dragStart[eye] = { x: x, y: y };
        }
    });

    canvas.addEventListener("mousemove", (evt) => {
        if (dragging[eye] === null) return;
        const [x, y] = toCanvasCoords(evt);
        const dragIdx = dragging[eye];
        
        if (dragIdx === 'all') {
            let dx = x - dragStart[eye].x;
            let dy = y - dragStart[eye].y;
            
            let pts = cropPoints[eye];
            let minX = Math.min(pts[0][0], pts[1][0], pts[2][0], pts[3][0]);
            let maxX = Math.max(pts[0][0], pts[1][0], pts[2][0], pts[3][0]);
            let minY = Math.min(pts[0][1], pts[1][1], pts[2][1], pts[3][1]);
            let Math_maxY = Math.max(pts[0][1], pts[1][1], pts[2][1], pts[3][1]);
            
            if (minX + dx < 0) dx = -minX;
            if (maxX + dx > 400) dx = 400 - maxX;
            if (minY + dy < 0) dy = -minY;
            if (Math_maxY + dy > 400) dy = 400 - Math_maxY;
            
            for (let i = 0; i < 4; i++) {
                pts[i][0] += dx;
                pts[i][1] += dy;
            }
            
            dragStart[eye] = { x: dragStart[eye].x + dx, y: dragStart[eye].y + dy };
            drawCanvas(eye);
            return;
        }

        if (getCropMode() === 'rectangular') {
            let pts = cropPoints[eye];
            let oppIdx = (dragIdx + 2) % 4; 
            let oppX = pts[oppIdx][0];
            let oppY = pts[oppIdx][1];
            
            let side = Math.max(Math.abs(x - oppX), Math.abs(y - oppY));
            
            let maxSideX = oppX < x ? 400 - oppX : oppX;
            let maxSideY = oppY < y ? 400 - oppY : oppY;
            side = Math.min(side, maxSideX, maxSideY);
            
            let newX = oppX + (side * (x > oppX ? 1 : -1));
            let newY = oppY + (side * (y > oppY ? 1 : -1));
            
            if (dragIdx === 0) { 
                pts[0] = [newX, newY]; pts[1] = [oppX, newY]; pts[2] = [oppX, oppY]; pts[3] = [newX, oppY];
            } else if (dragIdx === 1) { 
                pts[0] = [oppX, newY]; pts[1] = [newX, newY]; pts[2] = [newX, oppY]; pts[3] = [oppX, oppY];
            } else if (dragIdx === 2) { 
                pts[0] = [oppX, oppY]; pts[1] = [newX, oppY]; pts[2] = [newX, newY]; pts[3] = [oppX, newY];
            } else if (dragIdx === 3) { 
                pts[0] = [newX, oppY]; pts[1] = [oppX, oppY]; pts[2] = [oppX, newY]; pts[3] = [newX, newY];
            }
        } else {
            cropPoints[eye][dragIdx] = [x, y];
        }
        drawCanvas(eye);
    });

    window.addEventListener("mouseup", () => {
        if (dragging[eye] !== null) {
            dragging[eye] = null;
            fetch(`${API}/crop`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ eye: eye, points: cropPoints[eye] }) });
        }
    });
    drawCanvas(eye);
}

function setupStreamRetry(imgId, src) {
    const img = document.getElementById(imgId);
    if (!img) return;
    img.onerror = () => {
        img.src = "";
        setTimeout(() => { img.src = src + "?t=" + Date.now(); }, 2000);
    };
    setTimeout(() => { img.src = src + "?t=" + Date.now(); }, 3000);
}

function executeCmd(action) {
    fetch(`${API}/${action}`, { method: "POST" })
        .then((r) => r.json())
        .then((data) => { 
            if (!data.ok && data.error) alert(data.error); 
            if (data.crop_points) {
                cropPoints = data.crop_points;
                drawCanvas('L');
                drawCanvas('R');
            }
        });
}

function syncCrop(sourceEye) {
    fetch(`${API}/sync_crop/${sourceEye}`, { method: 'POST' })
        .then(r => r.json())
        .then(data => {
            if (data.ok && data.crop_points) {
                cropPoints.L = data.crop_points.L;
                cropPoints.R = data.crop_points.R;
                drawCanvas('L');
                drawCanvas('R');
            }
        });
}

function resetCrop() {
    cropPoints = JSON.parse(JSON.stringify(defaultCrop));
    drawCanvas('L');
    drawCanvas('R');
    
    fetch(`${API}/crop`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ eye: 'L', points: cropPoints.L }) });
    fetch(`${API}/crop`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ eye: 'R', points: cropPoints.R }) });
}

function closeInferenceModal() {
    document.getElementById('inference-modal').classList.add('modal-hidden');
    startAnotherTrial(); 
}

function viewDiagnosticReport() {
    if (window.latestReportUrl) {
        window.open(window.latestReportUrl, '_blank');
    }
    closeInferenceModal();
}

function submitInferenceSave() {
    const reqFields = ['dr-name', 'dr-license', 'patient-name', 'patient-dob', 'patient-sex'];
    let valid = true;
    
    reqFields.forEach(id => {
        let el = document.getElementById(id);
        if(!el.value) {
            el.style.borderColor = '#ef4444'; 
            valid = false;
        } else {
            el.style.borderColor = 'rgba(0, 210, 255, 0.3)';
        }
    });

    if(!valid) {
        alert("Please fill out all required fields marked with an asterisk (*).");
        return;
    }
    document.getElementById('patient-details-modal').classList.add('modal-hidden');
}

function executeFinalSave() {
    let diagnosisType = document.getElementById('diagnosis').value;
    let detailsPayload = {};

    if (diagnosisType === "UNKNOWN") {
        const reqFields = ['dr-name', 'dr-license', 'patient-name', 'patient-dob', 'patient-sex'];
        let valid = true;
        reqFields.forEach(id => { if(!document.getElementById(id).value) valid = false; });

        if(!valid) {
            alert("Please complete the Clinical Intake Details before generating a report.");
            document.getElementById('patient-details-modal').classList.remove('modal-hidden');
            return;
        }

        detailsPayload = {
            dr_name: document.getElementById('dr-name').value,
            dr_license: document.getElementById('dr-license').value,
            name: document.getElementById('patient-name').value,
            dob: document.getElementById('patient-dob').value,
            sex: document.getElementById('patient-sex').value,
            status: document.getElementById('patient-status').value,
            hba1c: document.getElementById('patient-hba1c').value,
            bp: document.getElementById('patient-bp').value
        };
    }

    let btnMain = document.getElementById('btn-save');
    let btnPopup = document.getElementById('btn-popup-save');
    if (btnMain) { btnMain.disabled = true; btnMain.innerHTML = "<span>[V]</span>SAVING..."; }
    if (btnPopup) { btnPopup.disabled = true; btnPopup.innerText = "SAVING..."; }

    fetch(`/api/recorder/save_trial`, { 
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(detailsPayload)
    })
        .then(r => r.json())
        .then(data => {
            if (data.ok) {
                let modal = document.getElementById('completion-modal');
                if(modal) modal.classList.add('modal-hidden');
                
                let btnNew = document.getElementById('btn-new-trial');
                if (btnMain) { 
                    btnMain.innerHTML = "<span>[V]</span>SAVED"; 
                    btnMain.style.borderColor = "#4ade80";
                    btnMain.style.color = "#4ade80";
                }
                if (btnPopup) { 
                    btnPopup.innerText = "DATA SAVED ✓"; 
                    btnPopup.style.borderColor = "#4ade80"; 
                }
                if (btnNew) {
                    btnNew.disabled = false;
                    btnNew.classList.add("pulse-highlight");
                }

                if (data.inference) {
                    document.getElementById('inference-result').innerText = data.inference.prediction;
                    document.getElementById('inference-confidence').innerText = data.inference.confidence;
                    document.getElementById('inference-modal').classList.remove('modal-hidden');
                    window.latestReportUrl = data.inference.report_url;
                }

            } else {
                alert("Error saving: " + data.error);
                if (btnMain) { btnMain.disabled = false; btnMain.innerHTML = "<span>[V]</span>SAVE"; }
                if (btnPopup) { btnPopup.disabled = false; btnPopup.innerText = diagnosisType === "UNKNOWN" ? "GENERATE REPORT" : "SAVE DATA"; }
            }
        });
}

function cmd(action) {
    if (action === "start") {
        resetCharts();
        sessionHistory = { L: [], R: [] }; 
        const pid = document.getElementById("patient-id").value.trim();
        fetch(`${API}/patient_id`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ patient_id: pid })
        }).then(() => executeCmd(action));
    } else if (action === "save") {
        executeFinalSave();
    
    // Intercept either trigger your HTML buttons might be sending
    } else if (action === "new_trial" || action === "discard") {
        startAnotherTrial(); 
    } else {
        executeCmd(action);
    }
}


function startAnotherTrial() {
    // 1. Instantly hide any popups
    let modal = document.getElementById('completion-modal');
    if (modal) {
        modal.classList.add('modal-hidden');
        modal.classList.remove('active-modal');
    }
    
    // 2. Unconditionally wipe the configuration panel using fallback logic
    let pidEl = document.getElementById('patient-id');
    if (pidEl) {
        pidEl.value = "";
        pidEl.style.borderColor = '';
        pidEl.style.boxShadow = '';
        pidEl.dispatchEvent(new Event('input'));
        pidEl.dispatchEvent(new Event('change'));
    }
    
    let diagEl = document.getElementById('diagnosis');
    if (diagEl) {
        diagEl.selectedIndex = 0;
        diagEl.dispatchEvent(new Event('change'));
    }
    
    // 3. Wipe any clinical intake details
    const formIds = ['dr-name', 'dr-license', 'patient-name', 'patient-dob', 'patient-sex', 'patient-status', 'patient-hba1c', 'patient-bp'];
    formIds.forEach(id => {
        let el = document.getElementById(id);
        if(el) { el.value = ""; el.style.borderColor = 'rgba(0, 210, 255, 0.3)'; }
    });
    
    // 4. Reset the action buttons back to default
    let btnMain = document.getElementById('btn-save');
    let btnPopup = document.getElementById('btn-popup-save');
    let btnNew = document.getElementById('btn-new-trial');
    
    if (btnMain) { 
        btnMain.innerHTML = "<span>[V]</span>SAVE"; 
        btnMain.style.borderColor = ""; 
        btnMain.style.color = ""; 
    }
    if (btnPopup) { 
        btnPopup.disabled = false; 
        btnPopup.innerText = "SAVE DATA"; 
        btnPopup.style.borderColor = ""; 
    }
    if (btnNew) {
        btnNew.disabled = true;
        btnNew.classList.remove("pulse-highlight");
    }
    
    resetCharts();
    sessionHistory = { L: [], R: [] }; 
    pollStatus();

    // 5. Tell the backend to quietly erase its internal flags
    fetch(`${API}/discard`, { method: "POST" }).catch(console.error);
}

function toggleFlip(eye, axis) { fetch(`${API}/transform`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ eye: eye, axis: axis }) }); }
function toggleRotate(eye) { fetch(`${API}/transform`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ eye: eye, rotate: true }) }); }

document.getElementById("patient-id").addEventListener("input", (e) => {
    const diag = document.getElementById("diagnosis").value;
    const isTracker = diag === "RITNET_TRACKER";
    const val = e.target.value.trim();
    
    if (isTracker) {
        if (val.length === 0 || /[^a-zA-Z0-9_-]/.test(val)) {
            e.target.style.borderColor = 'red'; e.target.style.boxShadow = '0 0 5px rgba(255,0,0,0.5)';
        } else {
            e.target.style.borderColor = ''; e.target.style.boxShadow = '';
        }
    } else {
        if (!/^\d{6}$/.test(val) && val.length > 0) {
            e.target.style.borderColor = 'red'; e.target.style.boxShadow = '0 0 5px rgba(255,0,0,0.5)';
        } else {
            e.target.style.borderColor = ''; e.target.style.boxShadow = '';
            if (/^\d{6}$/.test(val)) {
                e.target.blur();
            }
        }
    }
});

document.getElementById("patient-id").addEventListener("keydown", (e) => {
    if (e.key === "Enter") {
        e.preventDefault();
        document.getElementById('btn-start').click();
        e.target.blur();
    }
});

document.getElementById("patient-id").addEventListener("change", (e) => {
    fetch(`${API}/patient_id`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ patient_id: e.target.value.trim() }) });
});

document.getElementById("diagnosis").addEventListener("change", (e) => {
    const val = e.target.value;
    const pidInput = document.getElementById("patient-id");
    
    if (val === "RITNET_TRACKER") {
        pidInput.placeholder = "Enter Filename (e.g. subject_01)";
        pidInput.dispatchEvent(new Event('input'));
    } else {
        pidInput.placeholder = "Enter 6-digit Patient ID";
        pidInput.dispatchEvent(new Event('input'));
    }
    
    fetch(`${API}/diagnosis`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ diagnosis: val }) });
    e.target.blur(); 
    
    let btnMain = document.getElementById('btn-save');
    let btnPopup = document.getElementById('btn-popup-save');

    if (val === "UNKNOWN") {
        document.getElementById('patient-details-modal').classList.remove('modal-hidden');
        if (btnMain) btnMain.innerHTML = "<span>[V]</span>SAVE & REPORT";
        if (btnPopup) btnPopup.innerText = "GENERATE REPORT";
    } else {
        if (btnMain) btnMain.innerHTML = "<span>[V]</span>SAVE";
        if (btnPopup) btnPopup.innerText = "SAVE DATA";
    }
});

document.getElementById("chk-mp4").addEventListener("change", (e) => {
    fetch(`${API}/save_mp4`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ save_mp4: e.target.checked }) });
    e.target.blur(); 
});

function syncHardware() { 
    fetch(`${API}/reconnect_cameras`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ left_index: parseInt(document.getElementById('leftPort').value), right_index: parseInt(document.getElementById('rightPort').value) }) }); 
}

['leftPort', 'rightPort'].forEach(id => {
    let el = document.getElementById(id);
    if(el) el.addEventListener('change', (e) => e.target.blur());
});

['enableLeft', 'enableRight'].forEach(id => {
    document.getElementById(id).addEventListener('change', (e) => {
        fetch(`${API}/enabled`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ left_enabled: document.getElementById('enableLeft').checked, right_enabled: document.getElementById('enableRight').checked }) });
        e.target.blur(); 
        
        const eye = id === 'enableLeft' ? 'L' : 'R';
        applyDisabledOverlay(eye, e.target.checked);
    });
});

document.addEventListener('keydown', (e) => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    switch(e.key.toLowerCase()) {
        case 's': document.getElementById('btn-start').click(); break;
        case 'p': document.getElementById('btn-pause').click(); break;
        case 't': document.getElementById('btn-stop').click(); break;
        case 'd': document.getElementById('btn-discard').click(); break;
        case 'v': document.getElementById('btn-save').click(); break;
        case 'n': if(!document.getElementById('btn-new-trial').disabled) startAnotherTrial(); break;
    }
});

class MiniChart {
    constructor(canvasId, color, minX, maxX) {
        this.canvas = document.getElementById(canvasId);
        this.ctx = this.canvas.getContext('2d');
        this.color = color;
        this.data = [];
        this.minX = minX;
        this.maxX = maxX;
        this.minY = 1.0; 
        this.maxY = 9.0;
    }
    
    clear() { this.data = []; this.draw(); }
    
    addPoint(x, y, autoDraw = true) { 
        this.data.push({x, y}); 
        if (autoDraw) this.draw(); 
    }
    
    draw() {
        const w = this.canvas.width, h = this.canvas.height;
        this.ctx.clearRect(0, 0, w, h);
        
        const padY = 15;
        const usableH = h - (padY * 2);
        
        this.ctx.strokeStyle = 'rgba(255,255,255,0.1)';
        this.ctx.lineWidth = 1;
        this.ctx.beginPath();
        for(let i=1; i<=8; i++) {
            let yPos = padY + usableH - ((i - this.minY) / (this.maxY - this.minY)) * usableH;
            this.ctx.moveTo(0, yPos); this.ctx.lineTo(w, yPos);
            this.ctx.fillStyle = 'rgba(255,255,255,0.4)';
            this.ctx.font = '11px Arial'; 
            this.ctx.fillText(i + 'mm', 5, yPos - 4);
        }
        this.ctx.stroke();
        
        if(this.data.length === 0) return;
        
        this.ctx.strokeStyle = this.color;
        this.ctx.lineWidth = 3;
        this.ctx.beginPath();
        let hasStarted = false;
        
        this.data.forEach((pt) => {
            if (pt.y !== null) {
                let px = ((pt.x - this.minX) / (this.maxX - this.minX)) * w;
                let clampedY = Math.max(this.minY, Math.min(this.maxY, pt.y));
                let py = padY + usableH - ((clampedY - this.minY) / (this.maxY - this.minY)) * usableH;
                
                if (!hasStarted) {
                    this.ctx.moveTo(px, py);
                    hasStarted = true;
                } else {
                    this.ctx.lineTo(px, py); 
                }
            }
        });
        this.ctx.stroke();

        this.ctx.fillStyle = '#f59e0b'; 
        this.data.forEach((pt) => {
            if (pt.y === null) {
                let px = ((pt.x - this.minX) / (this.maxX - this.minX)) * w;
                this.ctx.fillRect(px - 2, h - padY + 4, 4, 8);
            }
        });
    }
}

function initCharts() {
    redChartInstance  = new MiniChart('chartRed',  '#ef4444', 0,   60);
    // UPDATED: Shifted X-axis constraints for Blue chart to [120, 180]
    blueChartInstance = new MiniChart('chartBlue', '#3b82f6', 120, 180);
    resetCharts();

    function fitCanvases() {
        ['chartRed', 'chartBlue'].forEach(id => {
            const c = document.getElementById(id);
            if (!c || !c.parentElement) return;
            const box = c.parentElement.getBoundingClientRect();
            if (box.width > 10 && box.height > 10) {
                c.width  = Math.floor(box.width);
                c.height = Math.floor(box.height);
            }
        });
        if (redChartInstance)  redChartInstance.draw();
        if (blueChartInstance) blueChartInstance.draw();
    }

    fitCanvases();
    window.addEventListener('resize', fitCanvases);
}

function resetCharts() {
    if (redChartInstance)  redChartInstance.clear();
    if (blueChartInstance) blueChartInstance.clear();
}

// UPDATED: Evaluates using the new 120-180 blue window
function updateCharts(elapsed, liveDiam) {
    if (document.getElementById("diagnosis").value === "RITNET_TRACKER") return;
    if (elapsed <= 60 && redChartInstance)                                redChartInstance.addPoint(elapsed, liveDiam);
    else if (elapsed >= 120 && elapsed <= 180 && blueChartInstance) blueChartInstance.addPoint(elapsed, liveDiam);
}

// UPDATED: Maps session history to the new 120-180 blue window 
function openLabView(eye) {
    activeLabEye = eye;
    resetCharts();
    
    fetch(`${API}/lab_view`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ active: true }) });
    
    const isTracker = document.getElementById("diagnosis").value === "RITNET_TRACKER";
    
    if (sessionHistory[eye] && !isTracker) { 
        sessionHistory[eye].forEach(pt => {
            if (pt.x <= 60 && redChartInstance) redChartInstance.addPoint(pt.x, pt.y, false);
            else if (pt.x >= 120 && pt.x <= 180 && blueChartInstance) blueChartInstance.addPoint(pt.x, pt.y, false);
        });
        if (redChartInstance) redChartInstance.draw();
        if (blueChartInstance) blueChartInstance.draw();
    }

    document.getElementById('lab-title').innerText = `TANGLAWAN Lab Suite - ${eye === 'L' ? 'LEFT' : 'RIGHT'} Eye Analysis`;
    document.getElementById('lab-video-stream').src = `/api/recorder/stream/crop/${eye}?t=${new Date().getTime()}`;
    document.getElementById('lab-view-modal').classList.remove('modal-hidden');
    pollStatus();
}

function closeLabView() {
    activeLabEye = null;
    document.getElementById('lab-view-modal').classList.add('modal-hidden');
    document.getElementById('lab-video-stream').src = "";
    
    fetch(`${API}/lab_view`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ active: false }) });
}

const METRIC_LABELS = [
    "Baseline Diam. [D_base] (mm)", "Minimum Diam. [D_min] (mm)",
    "Constriction Amplitude (mm)", "Constriction Latency (ms)",
    "Max Constrict. Velocity (mm/s)", "Avg Constrict. Velocity (mm/s)",
    "PIPR (mm)", "Max Redilation Velocity (mm/s)", "75% Recovery Time (ms)"
];

function updateLabViewTable(clinicalData, liveData) {
    const isTracker = document.getElementById("diagnosis").value === "RITNET_TRACKER";
    const tbody = document.getElementById('lab-table-body');
    tbody.innerHTML = '';
    
    METRIC_LABELS.forEach((label, index) => {
        const tr = document.createElement('tr');
        tr.innerHTML = `<td>${label}</td><td>${isTracker ? "--" : clinicalData.red[index]}</td><td>${isTracker ? "--" : clinicalData.blue[index]}</td>`;
        tbody.appendChild(tr);
    });
    
    document.getElementById('lab-fps').textContent  = liveData.fps.toFixed(1);
    
    let uiDiam = (isNaN(liveData.diameter_mm) || isTracker) ? "--" : liveData.diameter_mm.toFixed(2);
    document.getElementById('lab-diam').textContent = uiDiam;
    document.getElementById('lab-conf').textContent = isTracker ? "--" : liveData.confidence.toFixed(2);
}

function getValidDiam(liveData) {
    if (!liveData) return null;
    let y = parseFloat(liveData.diameter_mm);
    if (isNaN(y) || y <= 0 || liveData.confidence < 0.5) return null;
    return y;
}

// UPDATED: Dynamically updates timer text mapping exactly to the 3-minute max time
function pollStatus() {
    fetch(`${API}/status`)
        .then((r) => r.json())
        .then((data) => {
            const diag = document.getElementById("diagnosis").value;
            // 180 seconds for clinical protocol
            const maxTime = diag === "RITNET_TRACKER" ? 60 : 180;
            const maxTimeStr = diag === "RITNET_TRACKER" ? "01:00" : "03:00";
            
            let timerStr = `${String(Math.floor(data.elapsed / 60)).padStart(2, "0")}:${String(data.elapsed % 60).padStart(2, "0")} / ${maxTimeStr}`;
            
            const completionText = document.getElementById("completion-text");
            if (completionText) {
                completionText.innerText = diag === "RITNET_TRACKER" 
                    ? "The 1-minute raw dataset recording has finished." 
                    : "The 3-minute PLR diagnostic sequence has finished.";
            }

            document.getElementById("rec-status-badge").textContent = data.phase;
            document.getElementById("rec-timer").textContent = timerStr;
            
            let labPhase = document.getElementById("lab-proto-phase");
            if(labPhase) labPhase.textContent = data.phase;
            let labTimer = document.getElementById("lab-proto-timer");
            if(labTimer) labTimer.textContent = timerStr;

            document.getElementById("btn-save").disabled    = !(data.has_staged_data && !data.is_recording);
            document.getElementById("btn-discard").disabled = !data.has_staged_data;

            if (data.is_recording && !data.is_paused) {
                let valL = getValidDiam(data.live.L);
                let lastL = sessionHistory.L[sessionHistory.L.length - 1];
                if (!lastL || lastL.x !== data.elapsed) sessionHistory.L.push({ x: data.elapsed, y: valL });

                let valR = getValidDiam(data.live.R);
                let lastR = sessionHistory.R[sessionHistory.R.length - 1];
                if (!lastR || lastR.x !== data.elapsed) sessionHistory.R.push({ x: data.elapsed, y: valR });

                if (activeLabEye !== null && data.live[activeLabEye]) {
                    updateCharts(data.elapsed, getValidDiam(data.live[activeLabEye]));
                }
            }

            if (activeLabEye !== null && data.clinical[activeLabEye] && data.live[activeLabEye]) {
                updateLabViewTable(data.clinical[activeLabEye], data.live[activeLabEye]);
            }
            
            let modal = document.getElementById('completion-modal');
            if (modal && data.elapsed >= maxTime && data.has_staged_data && !modal.classList.contains('active-modal')) {
                
                if (data.is_recording) {
                    fetch(`${API}/stop`, { method: "POST" });
                }

                modal.classList.remove('modal-hidden');
                modal.classList.add('active-modal');
            }
        })
        .catch(console.error);
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

window.addEventListener("load", () => {
    setupCanvas("L");
    setupCanvas("R");
    initCharts();

    const leftChk = document.getElementById('enableLeft');
    const rightChk = document.getElementById('enableRight');
    if (leftChk) applyDisabledOverlay('L', leftChk.checked);
    if (rightChk) applyDisabledOverlay('R', rightChk.checked);

    fetch(`${API}/ping`, { method: "POST" });
    setInterval(() => fetch(`${API}/ping`, { method: "POST" }), 5000);

    setupStreamRetry("raw-L",  `${API}/stream/raw/L`);
    setupStreamRetry("raw-R",  `${API}/stream/raw/R`);
    setupStreamRetry("crop-L", `${API}/stream/crop/L`);
    setupStreamRetry("crop-R", `${API}/stream/crop/R`);

    setInterval(pollStatus,  500);
    setInterval(pollBattery, 3000);
    pollStatus();
    pollBattery();
    resetCrop();
});