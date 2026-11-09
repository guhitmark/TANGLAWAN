import os
import base64
import io
import math
import pdfkit
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def generate_diagnostic_report(state, patient_details, prediction, confidence):
    # Match the pathing logic exactly from exporters.py
    date_str = datetime.now().strftime("%m%d%Y") 
    patient_id = state.patient_id if state.patient_id else "UNKNOWN"
    trial_folder = f"{patient_id}_{date_str}"
    
    dataset_root = "/home/tanglawan/Desktop/vision_capture_v1/dataset"
    reports_dir = os.path.join(dataset_root, "UNKNOWN", trial_folder)
    os.makedirs(reports_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%B %d, %Y - %I:%M %p")
    file_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    report_filename = f"Diagnostic_Report_{patient_id}_{file_time}.html"
    pdf_filename = f"Diagnostic_Report_{patient_id}_{file_time}.pdf"
    
    report_path = os.path.join(reports_dir, report_filename)
    pdf_path = os.path.join(reports_dir, pdf_filename)

    # Generate Waveform Plot - Split into two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4), sharey=True, gridspec_kw={'width_ratios': [1, 1]})
    
    MAX_VALID_DROP = 0.5 
    
    for eye, color in [("L", "red"), ("R", "blue")]:
        data = state.session_data.get(eye, [])
        if data:
            t_red, y_red = [], []
            t_blue, y_blue = [], []
            
            last_valid_red = float('nan')
            last_valid_blue = float('nan')
            
            for d in data:
                t = d["t"]
                y = d["y"]
                is_valid = d["valid"] and y is not None and not math.isnan(y) and y > 1.0
                
                # Plot Red Phase
                if t <= 60:
                    if is_valid and not math.isnan(last_valid_red):
                        if (last_valid_red - y) > MAX_VALID_DROP:
                            is_valid = False 
                    if is_valid:
                        last_valid_red = y
                    if not math.isnan(last_valid_red):
                        t_red.append(t)
                        y_red.append(last_valid_red)
                        
                # Plot Blue Phase (120s to 180s)
                elif 120 <= t <= 180:
                    if is_valid and not math.isnan(last_valid_blue):
                        if (last_valid_blue - y) > MAX_VALID_DROP:
                            is_valid = False 
                    if is_valid:
                        last_valid_blue = y
                    if not math.isnan(last_valid_blue):
                        t_blue.append(t)
                        y_blue.append(last_valid_blue)
                    
            if t_red:
                ax1.plot(t_red, y_red, label=f"{eye} Eye" if eye == "L" else "", color=color, linewidth=1.0)
            if t_blue:
                ax2.plot(t_blue, y_blue, label=f"{eye} Eye" if eye == "R" else "", color=color, linewidth=1.0)

    # Format Left Subplot (Red Phase)
    ax1.set_title("Red Stimulus (0-60s)")
    ax1.set_xlim(0, 60)
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Pupil Diameter (mm)")
    ax1.set_ylim(1.5, 8.5)
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Format Right Subplot (Blue Phase)
    ax2.set_title("Blue Stimulus (120-180s)")
    ax2.set_xlim(120, 180)
    ax2.set_xlabel("Time (s)")
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    fig.suptitle("Pupillary Light Reflex (PLR) Waveform", fontsize=14, fontweight='bold')
    fig.legend(loc="upper right", bbox_to_anchor=(0.98, 0.95))
    
    plt.tight_layout()
    fig.subplots_adjust(top=0.82) 
    
    buf = io.BytesIO()
    plt.savefig(buf, format='png', dpi=150)
    plt.close()
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode('utf-8')
    
    # Extract Patient Data
    dr_name = patient_details.get("dr_name", "N/A")
    dr_license = patient_details.get("dr_license", "N/A")
    p_name = patient_details.get("name", "N/A")
    p_dob = patient_details.get("dob", "N/A")
    p_sex = patient_details.get("sex", "N/A")
    p_status = patient_details.get("status", "") or "Not Specified"
    p_hba1c = patient_details.get("hba1c", "") or "Not Tested"
    p_bp = patient_details.get("bp", "") or "Not Recorded"
    
    # -------------------------------------------------------------
    # HTML TEMPLATE WITH BROWSER-SIDE html2pdf.js INTEGRATION
    # -------------------------------------------------------------
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>TANGLAWAN Diagnostic Report - {patient_id}</title>
        <script src="https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js"></script>
        <style>
            body {{ 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; 
                background-color: #f8fafc; 
                padding: 20px; 
                margin: 0; 
            }}
            
            /* Strict box sizing to define exact PDF dimensions */
            #pdf-content {{
                background-color: #ffffff;
                width: 100%;
                max-width: 850px;
                margin: 0 auto;
                padding: 40px;
                box-sizing: border-box;
                border-radius: 8px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.05);
            }}

            .header {{ border-bottom: 3px solid #00d2ff; padding-bottom: 10px; margin-bottom: 20px; display: flex; justify-content: space-between; align-items: flex-end; }}
            .header h1 {{ color: #046b88; margin: 0; font-size: 1.6rem; }}
            .timestamp {{ color: #64748b; font-size: 0.85rem; font-weight: bold; }}
            
            .grid-container {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; margin-bottom: 20px; }}
            .info-box {{ background: #f8fafc; padding: 15px; border-radius: 8px; border: 1px solid #e2e8f0; }}
            .info-box h3 {{ margin-top: 0; color: #0f172a; border-bottom: 1px solid #cbd5e1; padding-bottom: 5px; font-size: 0.95rem; text-transform: uppercase; }}
            .info-table {{ width: 100%; text-align: left; font-size: 0.85rem; border-collapse: collapse; }}
            .info-table th {{ width: 120px; color: #64748b; font-weight: normal; padding: 4px 0; vertical-align: top; }}
            .info-table td {{ font-weight: 500; padding: 4px 0; }}
            
            .result-box {{ background: #eff6ff; padding: 15px; border-radius: 8px; border-left: 5px solid #3b82f6; margin-bottom: 20px; text-align: center; }}
            .result-title {{ font-size: 0.95rem; font-weight: bold; color: #1e3a8a; margin-bottom: 5px; text-transform: uppercase; }}
            .result-value {{ font-size: 2rem; font-weight: bold; color: #1d4ed8; letter-spacing: 1px; }}
            
            .graph-box {{ text-align: center; margin-top: 15px; }}
            .graph-box img {{ max-width: 100% !important; height: auto !important; border: 1px solid #cbd5e1; border-radius: 4px; }}
            
            .action-container {{ text-align: center; margin-top: 35px; padding-top: 25px; }}
            .btn {{ display: inline-block; padding: 12px 24px; font-size: 1rem; font-weight: bold; border-radius: 6px; cursor: pointer; border: none; margin: 0 10px; transition: 0.2s; }}
            .btn-print {{ background-color: #475569; color: #fff; }}
            .btn-print:hover {{ background-color: #334155; }}
            .btn-upload {{ background-color: #16a34a; color: #fff; }}
            .btn-upload:hover {{ background-color: #15803d; }}
            
            .footer {{ margin-top: 25px; font-size: 0.85rem; color: #94a3b8; text-align: center; }}
            
            @media print {{
                .action-container {{ display: none !important; }}
                body {{ padding: 0; background-color: #fff; }}
                #pdf-content {{ box-shadow: none; padding: 0; }}
            }}
        </style>
    </head>
    <body>
        
        <!-- Printable Wrapper Container -->
        <div id="pdf-content">
            <div class="header">
                <div>
                    <h1>TANGLAWAN Pupillometry System</h1>
                    <p style="margin: 5px 0 0 0; color: #475569;">Automated Clinical AI Diagnostic Report</p>
                </div>
                <div class="timestamp">Tested: {timestamp}</div>
            </div>
            
            <div class="grid-container">
                <div class="info-box">
                    <h3>Physician Information</h3>
                    <table class="info-table">
                        <tr><th>Ophthalmologist:</th><td>Dr. {dr_name}</td></tr>
                        <tr><th>License No:</th><td>{dr_license}</td></tr>
                        <tr><th>Device Node:</th><td>Jetson Nano Edge AI</td></tr>
                    </table>
                </div>
                
                <div class="info-box">
                    <h3>Patient Information</h3>
                    <table class="info-table">
                        <tr><th>Patient ID:</th><td><b>{patient_id}</b></td></tr>
                        <tr><th>Full Name:</th><td>{p_name}</td></tr>
                        <tr><th>Date of Birth:</th><td>{p_dob}</td></tr>
                        <tr><th>Sex:</th><td>{p_sex}</td></tr>
                    </table>
                </div>
            </div>
            
            <div class="info-box" style="margin-bottom: 20px;">
                <h3>Clinical Metrics</h3>
                <table class="info-table" style="display: grid; grid-template-columns: 1fr 1fr;">
                    <tbody>
                        <tr><th>General Status:</th><td>{p_status}</td></tr>
                        <tr><th>HbA1c Level:</th><td>{p_hba1c}</td></tr>
                        <tr><th>Blood Pressure:</th><td>{p_bp}</td></tr>
                    </tbody>
                </table>
            </div>
            
            <div class="result-box">
                <div class="result-title">1D-CNN Inference Result</div>
                <div class="result-value">{prediction}</div>
                <div style="margin-top: 10px; color: #475569; font-size: 1.1rem;">Diagnostic Confidence: <b style="color: #0f172a;">{confidence}%</b></div>
            </div>
            
            <div class="graph-box">
                <img src="data:image/png;base64,{img_b64}">
            </div>
        </div>
        
        <div class="action-container">
            <button class="btn btn-print" onclick="window.print()">🖨 Print to PDF</button>
            <button class="btn btn-upload" onclick="uploadData()">☁️ Auto-Upload to Supabase</button>
            
            <div class="footer">
                This report was generated automatically by the TANGLAWAN Edge Inference pipeline. <br>
                <i>Designed for research and screening purposes. Not a definitive medical diagnosis.</i>
            </div>
        </div>
        
        <script>
            async function uploadData() {{
                const btn = document.querySelector('.btn-upload');
                btn.innerText = "☁️ GENERATING & UPLOADING PDF...";
                btn.disabled = true;

                try {{
                    const SUPABASE_KEY = "sb_publishable_mIHOi8dllJ0BSou4PY1CZA_XjXCaYgG";
                    const pdfFilename = "{pdf_filename}";
                    
                    const element = document.getElementById('pdf-content');
                    const contentWidth = element.offsetWidth;
                    const contentHeight = element.offsetHeight + 20; // 20px buffer to prevent clipping

                    const opt = {
                        margin:       0,
                        filename:     pdfFilename,
                        image:        { type: 'jpeg', quality: 0.98 },
                        html2canvas:  { scale: 2, useCORS: true, windowWidth: contentWidth },
                        jsPDF:        { unit: 'px', format: [contentWidth, contentHeight], orientation: 'portrait' }
                    };

                    const pdfWorker = html2pdf().set(opt).from(element);
                    const pdfPdf = await pdfWorker.outputPdf('blob');
                    const pdfBlob = new Blob([pdfPdf], {{ type: 'application/pdf' }});

                    const STORAGE_URL = `https://zkmtfpeoihcxamtjatgh.supabase.co/storage/v1/object/tanglawan_pdfs/${{pdfFilename}}`;
                    
                    const uploadRes = await fetch(STORAGE_URL, {{
                        method: 'POST',
                        headers: {{
                            'apikey': SUPABASE_KEY,
                            'Authorization': `Bearer ${{SUPABASE_KEY}}`,
                            'Content-Type': 'application/pdf'
                        }},
                        body: pdfBlob
                    }});

                    if (!uploadRes.ok) {{
                        const errText = await uploadRes.text();
                        if (!errText.includes("Duplicate")) {{
                            alert("Storage Upload Failed: " + errText);
                            btn.innerText = "☁️ Auto-Upload to Supabase";
                            btn.disabled = false;
                            return;
                        }}
                    }}

                    const DB_URL = "https://zkmtfpeoihcxamtjatgh.supabase.co/rest/v1/tanglawan_reports";
                    const publicPdfUrl = `https://zkmtfpeoihcxamtjatgh.supabase.co/storage/v1/object/public/tanglawan_pdfs/${{pdfFilename}}`;
                    
                    const payload = {{
                        patient_id: "{patient_id}",
                        dr_name: "{dr_name}",
                        dr_license: "{dr_license}",
                        diagnosis: "{prediction}",
                        confidence: "{confidence}",
                        timestamp: "{file_time}",
                        report_html_b64: publicPdfUrl
                    }};

                    const dbRes = await fetch(DB_URL, {{
                        method: 'POST',
                        headers: {{
                            'Content-Type': 'application/json',
                            'apikey': SUPABASE_KEY,
                            'Authorization': `Bearer ${{SUPABASE_KEY}}`,
                            'Prefer': 'return=minimal'
                        }},
                        body: JSON.stringify(payload)
                    }});

                    if(dbRes.ok) {{
                        btn.innerText = "✓ DB & PDF SYNC SUCCESS";
                        btn.style.backgroundColor = "#16a34a";
                    }} else {{
                        const dbErr = await dbRes.text();
                        alert("Database Sync Failed: " + dbErr);
                        btn.innerText = "☁️ Auto-Upload to Supabase";
                        btn.disabled = false;
                    }}
                }} catch (error) {{
                    alert("Network Error: " + error);
                    btn.innerText = "☁️ Auto-Upload to Supabase";
                    btn.disabled = false;
                }}
            }}
        </script>
    </body>
    </html>
    """
    
    with open(report_path, "w") as f:
        f.write(html_content)
        
    options = {
        'page-width': '8.5in',
        'page-height': '13in',
        'margin-top': '0.4in',
        'margin-right': '0.4in',
        'margin-bottom': '0.4in',
        'margin-left': '0.4in',
        'encoding': "UTF-8",
        'enable-local-file-access': None,
        'no-outline': None
    }
    try:
        pdfkit.from_string(html_content, pdf_path, options=options)
    except Exception as e:
        print(f"PDF Generation Error: {e}")
        
    return report_path