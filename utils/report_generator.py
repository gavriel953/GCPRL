"""
PDF report generation for GCPRL enhancement results.
"""

import os
import io
import logging
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)


def generate_pdf_report(
    job_id: str,
    original_path: str,
    enhanced_path: str,
    metrics: dict,
    output_dir: str,
    method_name: str = 'GCPRL',
) -> str:
    """
    Generate a PDF report summarizing enhancement results.

    Args:
        job_id: Unique job identifier.
        original_path: Path to original image.
        enhanced_path: Path to enhanced image.
        metrics: Metrics dictionary.
        output_dir: Directory to save the PDF.
        method_name: Name of enhancement method.

    Returns:
        Filename of generated PDF.
    """
    try:
        from reportlab.lib.pagesizes import letter
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch
        from reportlab.lib import colors
        from reportlab.platypus import (
            SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
        )
        from reportlab.lib.enums import TA_CENTER
    except ImportError:
        raise ImportError("reportlab is required for PDF generation.")

    os.makedirs(output_dir, exist_ok=True)
    pdf_filename = f"report_{job_id}.pdf"
    pdf_path = os.path.join(output_dir, pdf_filename)

    doc = SimpleDocTemplate(pdf_path, pagesize=letter)
    styles = getSampleStyleSheet()
    story = []

    # Title
    title_style = ParagraphStyle('Title', parent=styles['Title'],
                                  fontSize=18, spaceAfter=12, alignment=TA_CENTER)
    story.append(Paragraph("GCPRL Medical Image Enhancement Report", title_style))
    story.append(Paragraph(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                           styles['Normal']))
    story.append(Paragraph(f"Job ID: {job_id}", styles['Normal']))
    story.append(Spacer(1, 0.2 * inch))

    # Metrics table
    story.append(Paragraph("Quality Metrics", styles['Heading2']))
    table_data = [['Metric', 'Original', 'Enhanced']]
    metric_rows = [
        ('CNR', metrics.get('cnr_original', 'N/A'), metrics.get('cnr_enhanced', 'N/A')),
        ('Entropy', metrics.get('entropy_original', 'N/A'), metrics.get('entropy_enhanced', 'N/A')),
        ('Edge Preservation', '-', metrics.get('edge_preservation', 'N/A')),
        ('Brightness Preservation', '-', metrics.get('brightness_preservation', 'N/A')),
        ('Processing Time (s)', '-', metrics.get('processing_time', 'N/A')),
    ]
    for row in metric_rows:
        table_data.append([str(x) for x in row])

    t = Table(table_data, colWidths=[3 * inch, 1.5 * inch, 1.5 * inch])
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1a3a5c')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f0f4f8')]),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#ccddee')),
    ]))
    story.append(t)
    story.append(Spacer(1, 0.3 * inch))

    # Images
    for label, path in [('Original Image', original_path), ('Enhanced Image', enhanced_path)]:
        if os.path.exists(path):
            story.append(Paragraph(label, styles['Heading3']))
            img = RLImage(path, width=4 * inch, height=3 * inch, kind='proportional')
            story.append(img)
            story.append(Spacer(1, 0.2 * inch))

    doc.build(story)
    logger.info(f"PDF report generated: {pdf_path}")
    return pdf_filename
