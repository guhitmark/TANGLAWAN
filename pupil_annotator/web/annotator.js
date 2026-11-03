const API = "/api/annotator";

let activeFolder = null;
let frames = [];
let selectedFrames = new Set();
let lastSelectedIdx = -1;

let ellipse = { cx: 200, cy: 200, rx: 0, ry: 0, angle: 0 };
let clipboardEllipse = null;

let isSelected = false;
let isDrawing = false;
let isRotating = false;
let isDragging = false;
let resizeMode = null; 
let dragOffset = { x: 0, y: 0 };

let undoStack = [];
let redoStack = [];

const canvas = document.getElementById("canvas");
const ctx = canvas.getContext("2d");
const img = new Image();
img.onload = () => drawCanvas();

window.addEventListener("load", () => {
    loadFolders();
    pollBattery();
    setInterval(pollBattery, 15000);
});

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

function updateProgress() {
    const total = frames.length;
    let labeled = 0;
    frames.forEach(f => { 
        const el = document.getElementById(f);
        if (el && el.classList.contains('labeled')) labeled++; 
    });
    
    document.getElementById("prog-total").innerText = total;
    document.getElementById("prog-labeled").innerText = labeled;
    document.getElementById("prog-unlabeled").innerText = total - labeled;
}

async function loadFolders() {
    const res = await fetch(`${API}/folders`, { cache: 'no-store' }).then(r => r.json());
    const container = document.getElementById("folder-container");
    container.innerHTML = "";

    if (res.folders.length === 0) {
        container.innerHTML = '<p style="color: #94a3b8; font-size:0.8rem; text-align:center;">No recordings found.</p>';
        return;
    }

    res.folders.forEach(motherObj => {
        const motherWrapper = document.createElement("div");
        motherWrapper.style.marginBottom = "10px";

        const motherHeader = document.createElement("div");
        motherHeader.className = "folder-header";
        motherHeader.innerHTML = `<span>📁 ${motherObj.mother_name}</span>`;
        motherHeader.style.backgroundColor = "rgba(4, 13, 26, 0.9)";
        motherHeader.style.borderColor = "rgba(0, 210, 255, 0.4)";
        
        const motherContent = document.createElement("div");
        motherContent.style.paddingLeft = "15px"; 
        motherContent.style.borderLeft = "1px solid rgba(0, 210, 255, 0.2)";
        motherContent.style.marginLeft = "10px";

        motherObj.trials.forEach(trial => {
            const trialHeader = document.createElement("div");
            trialHeader.className = "folder-header";
            trialHeader.style.marginTop = "5px";
            trialHeader.innerHTML = `<span>📂 ${trial.name}</span> <span>[${trial.frames.length}]</span>`;
            
            const trialContent = document.createElement("div");
            trialContent.className = "folder-content";

            if (trial.id === activeFolder) {
                trialHeader.classList.add("active");
                trialContent.classList.add("open");
            }

            trialHeader.onclick = () => {
                const isOpen = trialContent.classList.contains("open");
                document.querySelectorAll(".folder-content").forEach(el => el.classList.remove("open"));
                document.querySelectorAll(".folder-header").forEach(el => el.classList.remove("active"));
                
                if (!isOpen) {
                    trialHeader.classList.add("active");
                    trialContent.classList.add("open");
                    activeFolder = trial.id;
                    frames = trial.frames;
                } else {
                    activeFolder = null;
                    frames = [];
                }
                updateProgress();
            };

            if (trial.frames.length > 0) {
                trial.frames.forEach((f, idx) => {
                    const item = document.createElement("div");
                    item.className = "frame-item";
                    item.id = f;
                    item.innerText = `📄 ${f.replace('.jpg', '')}`; 
                    
                    if (trial.labeled.includes(f)) {
                        item.classList.add("labeled");
                    } else {
                        item.classList.add("unlabeled");
                    }
                    
                    item.onclick = (e) => {
                        frames = trial.frames; 
                        handleFrameClick(e, f, idx, trial.id, trial.frames);
                    };
                    trialContent.appendChild(item);
                });
                
                if (activeFolder === trial.id) {
                    frames = trial.frames;
                    updateProgress();
                }
            } else {
                trialContent.innerHTML = '<p style="color:#facc15; font-size:0.75rem; margin:0; padding-left: 10px;">No frames extracted yet. Click "Extract Selected".</p>';
            }

            motherContent.appendChild(trialHeader);
            motherContent.appendChild(trialContent);
        });

        motherWrapper.appendChild(motherHeader);
        motherWrapper.appendChild(motherContent);
        container.appendChild(motherWrapper);
    });
}

async function loadFrameAnnotation(folderName, f) {
    try {
        const res = await fetch(`${API}/dataset/${folderName}/label/${f}`).then(r => r.json());
        if (res.ok && res.ellipse) {
            ellipse = res.ellipse;
        } else {
            ellipse = { cx: 200, cy: 200, rx: 0, ry: 0, angle: 0 };
        }
    } catch (err) {
        ellipse = { cx: 200, cy: 200, rx: 0, ry: 0, angle: 0 };
    }
    isSelected = false;
    undoStack = [];
    redoStack = [];
    img.src = `${API}/dataset/${folderName}/frames/${f}?t=${new Date().getTime()}`;
}

async function handleFrameClick(e, f, idx, folderName, folderFrames) {
    if (e.shiftKey && lastSelectedIdx !== -1) {
        let start = Math.min(idx, lastSelectedIdx);
        let end = Math.max(idx, lastSelectedIdx);
        for(let i=start; i<=end; i++) selectedFrames.add(folderFrames[i]);
    } else if (e.ctrlKey) {
        if(selectedFrames.has(f)) selectedFrames.delete(f);
        else selectedFrames.add(f);
    } else {
        selectedFrames.clear();
        selectedFrames.add(f);
    }
    
    lastSelectedIdx = idx;
    document.getElementById("current-frame-lbl").innerText = `VIEWING: ${f.replace('.jpg', '')}`;
    
    await loadFrameAnnotation(folderName, f);
    
    document.querySelectorAll(".frame-item").forEach(el => {
        el.classList.toggle("selected", selectedFrames.has(el.id));
    });
}

async function extractActiveFolder() {
    if (!activeFolder) return alert("Select a trial folder (e.g. eye1_L) from the list first.");
    const count = document.getElementById("frame-count").value;
    document.getElementById('current-frame-lbl').innerText = `EXTRACTING ${activeFolder}...`;
    document.getElementById('current-frame-lbl').style.color = "#facc15";
    
    const res = await fetch(`${API}/extract`, { method: "POST", body: JSON.stringify({ trial_folder: activeFolder, frames: count }) }).then(r=>r.json());
    if(res.ok) { 
        document.getElementById('current-frame-lbl').innerText = "EXTRACTION COMPLETE.";
        document.getElementById('current-frame-lbl').style.color = "#4ade80";
        loadFolders(); 
    } else {
        alert("Error: " + res.error);
        document.getElementById('current-frame-lbl').innerText = "AWAITING SELECTION";
        document.getElementById('current-frame-lbl').style.color = "#00e5ff";
    }
}

async function exportDataset() {
    // Check if the currently viewed folder has missing frames to warn the user, but export proceeds globally
    const unlabeled = frames.filter(f => !document.getElementById(f).classList.contains("labeled"));
    if (frames.length > 0 && unlabeled.length > 0) {
        unlabeled.forEach(f => document.getElementById(f).classList.add("unlabeled-error"));
        const proceed = confirm(`⚠️ INCOMPLETE FOLDER DETECTED ⚠️\n\nYou have ${unlabeled.length} unannotated frames in your current view.\n\nDo you want to proceed and export ALL globally completed annotations anyway?`);
        if (!proceed) return;
    }

    frames.forEach(f => {
        const el = document.getElementById(f);
        if(el) el.classList.remove("unlabeled-error");
    });

    const res = await fetch(`${API}/export`, { method: "POST", body: JSON.stringify({}) }).then(r=>r.json());
    if (res.ok) alert(`Successfully compiled and exported ${res.count} total frames into Unified Train/Val/Test sets.`);
    else alert("Export Error: " + res.error);
}

async function discardDataset() {
    if (!activeFolder) return alert("Select a trial folder from the list first.");
    const confirmDiscard = confirm(`WARNING: Are you sure you want to delete all extracted frames and masks for '${activeFolder}'?\n\n(The original .avi video will NOT be deleted)`);
    if (confirmDiscard) {
        const res = await fetch(`${API}/discard`, { method: "POST", body: JSON.stringify({ trial_folder: activeFolder }) }).then(r=>r.json());
        if (res.ok) {
            img.src = ""; 
            drawCanvas();
            document.getElementById('current-frame-lbl').innerText = "AWAITING SELECTION";
            document.getElementById('current-frame-lbl').style.color = "#00e5ff";
            frames = [];
            selectedFrames.clear();
            
            setTimeout(() => {
                loadFolders();
                updateProgress();
            }, 200);
        } else alert("Error: " + res.error);
    }
}

async function saveAnnotation() {
    if(selectedFrames.size === 0) return;
    const targetFrames = Array.from(selectedFrames);
    
    const res = await fetch(`${API}/save_annotation`, { 
        method: "POST", 
        body: JSON.stringify({ trial: activeFolder, frames: targetFrames, ellipse: ellipse }) 
    }).then(r => r.json());
    
    targetFrames.forEach(f => {
        const el = document.getElementById(f);
        if(el) {
            if (res.status === "deleted") {
                el.classList.remove("labeled");
                el.classList.add("unlabeled");
            } else {
                el.classList.add("labeled");
                el.classList.remove("unlabeled");
            }
        }
    });
    updateProgress();
}

async function deleteSelectedFrames() {
    if (selectedFrames.size === 0 || !activeFolder) return;
    
    const targetFrames = Array.from(selectedFrames);
    const proceed = confirm(`Are you sure you want to permanently delete these ${targetFrames.length} frame(s)? This cannot be undone.`);
    if (!proceed) return;

    const res = await fetch(`${API}/delete_frames`, { 
        method: "POST", 
        body: JSON.stringify({ trial: activeFolder, frames: targetFrames }) 
    }).then(r => r.json());

    if (res.ok) {
        // Remove the deleted items from the local array and the DOM
        targetFrames.forEach(f => {
            frames = frames.filter(frame => frame !== f);
            const el = document.getElementById(f);
            if (el) el.remove();
        });

        selectedFrames.clear();
        
        // Auto-select the next available frame (or clear if empty)
        if (frames.length > 0) {
            let nextIdx = Math.min(lastSelectedIdx, frames.length - 1);
            let nextFrame = frames[nextIdx];
            
            selectedFrames.add(nextFrame);
            lastSelectedIdx = nextIdx;
            
            const nextEl = document.getElementById(nextFrame);
            if(nextEl) {
                nextEl.classList.add("selected");
                nextEl.scrollIntoView({block: "nearest", behavior: "smooth"});
            }
            
            await loadFrameAnnotation(activeFolder, nextFrame);
            document.getElementById("current-frame-lbl").innerText = `VIEWING: ${nextFrame.replace('.jpg', '')}`;
        } else {
            img.src = "";
            drawCanvas();
            document.getElementById("current-frame-lbl").innerText = "AWAITING SELECTION";
        }
        
        updateProgress();
    } else {
        alert("Error deleting frames: " + res.error);
    }
}

function saveState() {
    undoStack.push(JSON.parse(JSON.stringify(ellipse)));
    if (undoStack.length > 50) undoStack.shift();
    redoStack = [];
}

function undo() {
    if (undoStack.length > 0) {
        redoStack.push(JSON.parse(JSON.stringify(ellipse)));
        ellipse = undoStack.pop();
        drawCanvas(); saveAnnotation();
    }
}

function redo() {
    if (redoStack.length > 0) {
        undoStack.push(JSON.parse(JSON.stringify(ellipse)));
        ellipse = redoStack.pop();
        drawCanvas(); saveAnnotation();
    }
}

function getCanvasCoords(e) {
    const rect = canvas.getBoundingClientRect();
    const scaleX = canvas.width / rect.width;
    const scaleY = canvas.height / rect.height;
    return { x: (e.clientX - rect.left) * scaleX, y: (e.clientY - rect.top) * scaleY };
}

function isPointInEllipse(px, py) {
    if (ellipse.rx === 0 || ellipse.ry === 0) return false;
    const dx = px - ellipse.cx;
    const dy = py - ellipse.cy;
    const rad = -ellipse.angle * Math.PI / 180;
    const localX = dx * Math.cos(rad) - dy * Math.sin(rad);
    const localY = dx * Math.sin(rad) + dy * Math.cos(rad);
    return (localX*localX)/(ellipse.rx*ellipse.rx) + (localY*localY)/(ellipse.ry*ellipse.ry) <= 1;
}

function getClickedHandle(px, py) {
    if (!isSelected) return null;
    
    // FIXED: Give absolute priority to the center node if clicked within 8 pixels
    // Prevents perimeter resize boundaries from intercepting drags on tiny pupils
    if (Math.hypot(px - ellipse.cx, py - ellipse.cy) <= 8) return 'center';

    const rad = ellipse.angle * Math.PI / 180;
    const cos = Math.cos(rad), sin = Math.sin(rad);
    const handles = {
        rx1: { x: ellipse.cx + ellipse.rx * cos, y: ellipse.cy + ellipse.rx * sin, type: 'rx' },
        rx2: { x: ellipse.cx - ellipse.rx * cos, y: ellipse.cy - ellipse.rx * sin, type: 'rx' },
        ry1: { x: ellipse.cx - ellipse.ry * sin, y: ellipse.cy + ellipse.ry * cos, type: 'ry' },
        ry2: { x: ellipse.cx + ellipse.ry * sin, y: ellipse.cy - ellipse.ry * cos, type: 'ry' }
    };
    for (let key in handles) {
        if (Math.hypot(px - handles[key].x, py - handles[key].y) < 15) return handles[key].type;
    }
    return null;
}

let startX, startY;

canvas.onmousedown = (e) => {
    const pos = getCanvasCoords(e);
    if (e.button === 2) { saveState(); isRotating = true; return; }
    
    let clickedHandle = getClickedHandle(pos.x, pos.y);
    
    // Explicit routing for the protected center node
    if (clickedHandle === 'center') {
        saveState(); isDragging = true;
        dragOffset.x = pos.x - ellipse.cx;
        dragOffset.y = pos.y - ellipse.cy;
        canvas.style.cursor = 'grabbing';
        return;
    } else if (clickedHandle) { 
        saveState(); resizeMode = clickedHandle; 
        return; 
    }

    if (isPointInEllipse(pos.x, pos.y)) {
        if (!isSelected) { isSelected = true; drawCanvas(); } 
        else {
            saveState(); isDragging = true;
            dragOffset.x = pos.x - ellipse.cx;
            dragOffset.y = pos.y - ellipse.cy;
            canvas.style.cursor = 'grabbing';
        }
        return;
    }

    if (isSelected) {
        isSelected = false;
        drawCanvas();
        return;
    }

    isSelected = false; saveState();
    isDrawing = true; ellipse.angle = 0;
    startX = pos.x; startY = pos.y;
    canvas.style.cursor = 'crosshair';
};

canvas.onmousemove = (e) => {
    const pos = getCanvasCoords(e);
    
    if (isDrawing) {
        ellipse.cx = startX;
        ellipse.cy = startY;
        ellipse.rx = Math.abs(pos.x - startX);
        ellipse.ry = Math.abs(pos.y - startY);
        drawCanvas();
    } else if (isRotating) {
        ellipse.angle = Math.atan2(pos.y - ellipse.cy, pos.x - ellipse.cx) * (180 / Math.PI);
        canvas.style.cursor = 'alias';
        drawCanvas();
    } else if (isDragging) {
        ellipse.cx = pos.x - dragOffset.x;
        ellipse.cy = pos.y - dragOffset.y;
        drawCanvas();
    } else if (resizeMode) {
        const dist = Math.hypot(pos.x - ellipse.cx, pos.y - ellipse.cy);
        if (resizeMode === 'rx') ellipse.rx = dist;
        if (resizeMode === 'ry') ellipse.ry = dist;
        drawCanvas();
    } else {
        if (getClickedHandle(pos.x, pos.y)) canvas.style.cursor = 'pointer';
        else if (isPointInEllipse(pos.x, pos.y)) canvas.style.cursor = isSelected ? 'grab' : 'pointer';
        else canvas.style.cursor = 'crosshair';
    }
};

canvas.onmouseup = () => { 
    if (isDrawing || isRotating || isDragging || resizeMode) saveAnnotation(); 
    isDrawing = false; isRotating = false; isDragging = false; resizeMode = null;
    canvas.style.cursor = 'crosshair';
};

canvas.oncontextmenu = (e) => e.preventDefault();

function drawCanvas() {
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    if(img.src) ctx.drawImage(img, 0, 0, 400, 400);
    
    if(ellipse.rx > 0 && ellipse.ry > 0) {
        ctx.beginPath();
        ctx.ellipse(ellipse.cx, ellipse.cy, ellipse.rx, ellipse.ry, ellipse.angle * Math.PI/180, 0, 2 * Math.PI);
        
        ctx.strokeStyle = isSelected ? "rgba(250, 204, 21, 0.65)" : "#00e5ff";
        ctx.lineWidth = isSelected ? 1.5 : 1.0;
        ctx.stroke();
        
        if (isSelected) {
            const rad = ellipse.angle * Math.PI / 180;
            const cos = Math.cos(rad), sin = Math.sin(rad);
            
            ctx.fillStyle = "#ef4444";
            ctx.fillRect(ellipse.cx - 2, ellipse.cy - 2, 4, 4);
            
            ctx.fillStyle = "#facc15";
            const drawNode = (x, y) => ctx.fillRect(x - 3, y - 3, 6, 6);
            drawNode(ellipse.cx + ellipse.rx * cos, ellipse.cy + ellipse.rx * sin); 
            drawNode(ellipse.cx - ellipse.rx * cos, ellipse.cy - ellipse.rx * sin); 
            drawNode(ellipse.cx - ellipse.ry * sin, ellipse.cy + ellipse.ry * cos); 
            drawNode(ellipse.cx + ellipse.ry * sin, ellipse.cy - ellipse.ry * cos); 
        } else {
            ctx.fillStyle = "#4ade80";
            ctx.fillRect(ellipse.cx - 2, ellipse.cy - 2, 4, 4);
        }
    }
}

document.addEventListener("keydown", async (e) => {
    if (e.ctrlKey && e.key.toLowerCase() === 'c') clipboardEllipse = { ...ellipse };
    if (e.ctrlKey && e.key.toLowerCase() === 'v') {
        if (clipboardEllipse && selectedFrames.size > 0) {
            saveState();
            ellipse = { ...clipboardEllipse };
            drawCanvas(); saveAnnotation();
        }
    }
    if (e.ctrlKey && (e.key.toLowerCase() === 'z' || e.key.toLowerCase() === 'x')) undo();
    if (e.ctrlKey && e.key.toLowerCase() === 'y') redo();
    
    if ((e.key === 'Delete' || e.key === 'Backspace') && isSelected) {
        saveState();
        ellipse = { cx: 0, cy: 0, rx: 0, ry: 0, angle: 0 };
        isSelected = false;
        drawCanvas(); saveAnnotation();
    }

    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
        if (frames.length === 0 || lastSelectedIdx === -1) return;
        e.preventDefault(); 

        let nextIdx = e.key === "ArrowDown" ? lastSelectedIdx + 1 : lastSelectedIdx - 1;
        if (nextIdx >= 0 && nextIdx < frames.length) {
            let f = frames[nextIdx];
            
            if (e.shiftKey) selectedFrames.add(f);
            else {
                selectedFrames.clear();
                selectedFrames.add(f);
            }

            lastSelectedIdx = nextIdx;
            
            const el = document.getElementById(f);
            if (el) el.scrollIntoView({block: "nearest", behavior: "smooth"});
            
            await loadFrameAnnotation(activeFolder, f);
            document.getElementById("current-frame-lbl").innerText = `VIEWING: ${f.replace('.jpg', '')}`;
            
            document.querySelectorAll(".frame-item").forEach(item => {
                item.classList.toggle("selected", selectedFrames.has(item.id));
            });
        }
    }
});