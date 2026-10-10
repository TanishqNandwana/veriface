"""
VeriFace PDF Forensic Report Generator
Uses ReportLab to generate official PDF forensic reports for completed media analyses.
"""

import os
from datetime import datetime
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage

def generate_forensic_pdf_report(case_data: dict, output_pdf_path: str) -> str:
    """
    Generates a professional forensic PDF report for a given case_data dictionary.
    Returns absolute path of the generated PDF report.
    """
    os.makedirs(os.path.dirname(output_pdf_path), exist_ok=True)
    
    doc = SimpleDocTemplate(
        output_pdf_path,
        pagesize=letter,
        rightMargin=36,
        leftMargin=36,
        topMargin=36,
        bottomMargin=36
    )
    
    styles = getSampleStyleSheet()
    
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Heading1'],
        fontName='Helvetica-Bold',
        fontSize=22,
        textColor=colors.HexColor("#0f172a"),
        spaceAfter=6
    )
    
    subtitle_style = ParagraphStyle(
        'DocSubTitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=11,
        textColor=colors.HexColor("#64748b"),
        spaceAfter=15
    )
    
    heading_style = ParagraphStyle(
        'SectionHeading',
        parent=styles['Heading2'],
        fontName='Helvetica-Bold',
        fontSize=14,
        textColor=colors.HexColor("#1e293b"),
        spaceBefore=12,
        spaceAfter=6
    )
    
    body_style = ParagraphStyle(
        'BodyTextCustom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=10,
        textColor=colors.HexColor("#334155"),
        leading=14
    )

    story = []

    # Title & Branding
    story.append(Paragraph("VeriFace — Media Forensic Analysis Report", title_style))
    story.append(Paragraph("Verify the Media. Reveal the Truth. | Automated AI Forensic Platform", subtitle_style))
    story.append(Spacer(1, 10))

    # Case Summary Table
    verdict = str(case_data.get("verdict", "UNKNOWN")).upper()
    conf = case_data.get("confidence", 0.0) * 100
    
    verdict_color = colors.HexColor("#ef4444") if verdict == "MANIPULATED" else colors.HexColor("#10b981")
    
    summary_table_data = [
        [Paragraph("<b>Case ID:</b>", body_style), Paragraph(str(case_data.get("case_id")), body_style)],
        [Paragraph("<b>Filename:</b>", body_style), Paragraph(str(case_data.get("filename")), body_style)],
        [Paragraph("<b>Media Type:</b>", body_style), Paragraph(str(case_data.get("media_type")).capitalize(), body_style)],
        [Paragraph("<b>SHA-256 Hash:</b>", body_style), Paragraph(str(case_data.get("file_hash"))[:32] + "...", body_style)],
        [Paragraph("<b>Timestamp:</b>", body_style), Paragraph(str(case_data.get("created_at", datetime.now().isoformat())), body_style)],
        [Paragraph("<b>Final Verdict:</b>", body_style), Paragraph(f"<font color='{verdict_color.hexval()}'><b>{verdict}</b></font>", body_style)],
        [Paragraph("<b>Confidence Score:</b>", body_style), Paragraph(f"<b>{conf:.2f}%</b>", body_style)],
        [Paragraph("<b>Primary Model:</b>", body_style), Paragraph(str(case_data.get("model_name")), body_style)],
    ]
    
    t = Table(summary_table_data, colWidths=[140, 380])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#f8fafc")),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#e2e8f0")),
        ('PADDING', (0, 0), (-1, -1), 6),
    ]))
    story.append(t)
    story.append(Spacer(1, 15))

    # Forensic Details & Technical Evidence
    story.append(Paragraph("Forensic Evidence & Model Outputs", heading_style))
    
    details = case_data.get("analysis_details", {})
    explanation = details.get("explanation", {})
    desc = explanation.get("description", "Automated deepfake feature extraction.")
    story.append(Paragraph(f"<b>Attribution Technique:</b> {desc}", body_style))
    story.append(Spacer(1, 10))
    
    # Insert Heatmap Image if present
    overlay_path = explanation.get("overlay_path")
    if overlay_path and os.path.exists(overlay_path):
        try:
            img = RLImage(overlay_path, width=200, height=200)
            story.append(img)
            story.append(Paragraph("<i>Figure 1: Grad-CAM suspicious facial feature attribution heatmap overlay.</i>", body_style))
            story.append(Spacer(1, 10))
        except Exception:
            pass

    # Temporal / Segment Analysis Table
    if "temporal_timeline" in details:
        story.append(Paragraph("Frame-Level Temporal Timeline", heading_style))
        timeline = details["temporal_timeline"][:5]
        t_rows = [["Frame #", "Timestamp (s)", "Manipulation Score"]]
        for f in timeline:
            t_rows.append([str(f["frame_index"]), f"{f['timestamp_sec']}s", f"{f['score']*100:.1f}%"])
        tt = Table(t_rows, colWidths=[100, 150, 150])
        tt.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(tt)
        story.append(Spacer(1, 10))
        
    elif "segment_saliency" in details:
        story.append(Paragraph("Audio Segment Saliency Analysis", heading_style))
        segments = details["segment_saliency"][:5]
        s_rows = [["Segment #", "Time Interval (s)", "Synthetic Score"]]
        for seg in segments:
            s_rows.append([str(seg["segment_index"]), f"{seg['start_time_sec']}s - {seg['end_time_sec']}s", f"{seg['score']*100:.1f}%"])
        st = Table(s_rows, colWidths=[100, 150, 150])
        st.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e2e8f0")),
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#cbd5e1")),
            ('PADDING', (0, 0), (-1, -1), 4),
        ]))
        story.append(st)
        story.append(Spacer(1, 10))

    # Standards & Legal Disclaimer
    story.append(Spacer(1, 15))
    story.append(Paragraph("Standards & Forensic Disclaimer", heading_style))
    disclaimer_text = (
        "This report was automatically generated by the VeriFace Deepfake Forensic Detection Platform. "
        "Automated deepfake detection algorithms evaluate statistical feature distributions, frequency artifacts, "
        "and latent flow trajectories. Confidence scores reflect model-calibrated likelihoods and should be "
        "evaluated alongside expert human forensic review."
    )
    story.append(Paragraph(disclaimer_text, body_style))
    
    doc.build(story)
    return output_pdf_path

if __name__ == "__main__":
    dummy_case = {
        "case_id": "TEST_CASE_001",
        "filename": "sample_face.jpg",
        "media_type": "image",
        "file_hash": "a1b2c3d4e5f67890123456789abcdef0123456789abcdef0123456789abcdef0",
        "verdict": "manipulated",
        "confidence": 0.9452,
        "model_score": 0.9452,
        "model_name": "FA-ViT (Forgery-Aware Vision Transformer)",
        "analysis_details": {
            "explanation": {
                "description": "Grad-CAM heatmap highlighting facial patches with high forgery feature activations."
            }
        }
    }
    out = generate_forensic_pdf_report(dummy_case, "forensic_reports/TEST_CASE_001_report.pdf")
    print("PDF Generation Test passed! Output file:", out)
