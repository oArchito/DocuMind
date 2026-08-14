"""
generate_sample_pdf.py  –  Create a sample PDF for evaluation.

This script generates a PDF about a fictional company called "NovaTech Solutions"
containing 15+ distinct facts that we can test our RAG system against.

Run:
    python generate_sample_pdf.py
"""

import os
import sys

# We'll create a simple PDF using only the built-in capabilities
# by writing a minimal PDF file manually (no extra dependencies needed).

SAMPLE_TEXT = """NovaTech Solutions - Employee Handbook 2025

Chapter 1: Company Overview

NovaTech Solutions was founded in 2018 by Dr. Elena Vasquez and Marcus Chen in Austin, Texas. The company specialises in building AI-powered supply-chain optimisation tools for mid-size manufacturers. As of January 2025, NovaTech employs 342 people across four offices located in Austin, Toronto, Berlin, and Singapore.

The company mission statement is: "Empowering manufacturers with intelligent automation to reduce waste, cut costs, and deliver faster." NovaTech's flagship product is called OptiFlow, a cloud-based platform that uses machine learning to predict demand, optimise inventory levels, and route shipments efficiently. OptiFlow currently serves over 1,200 enterprise customers in 38 countries.

Chapter 2: Work Policies

All full-time employees receive 22 days of paid vacation per year, plus 10 public holidays. Vacation days increase to 27 days after five years of continuous employment. Sick leave is unlimited but requires a medical certificate for absences longer than three consecutive days.

Remote work is permitted up to three days per week for most roles. Employees must be available during core hours, which are 10:00 AM to 3:00 PM in their local time zone. Friday is designated as a "no-meetings" day to allow deep focus work.

The standard work week is 40 hours. Overtime must be pre-approved by a direct manager and is compensated at 1.5 times the normal hourly rate. Employees in engineering roles are eligible for on-call rotations, which include an additional stipend of $500 per on-call week.

Chapter 3: Benefits and Compensation

NovaTech offers a competitive benefits package. Health insurance is fully covered for employees and 75 percent covered for dependents through BlueCross Premium Plan. The company matches 401(k) contributions up to 6 percent of the employee's salary.

An annual learning budget of $3,000 is provided to each employee for courses, conferences, or certifications. Employees can also apply for a one-time educational grant of up to $15,000 for degree programmes related to their role.

The performance review cycle occurs twice a year, in March and September. Bonuses range from 5 percent to 20 percent of annual salary based on individual and company performance. The average bonus paid in 2024 was 12 percent.

Chapter 4: Technology and Security

All employees are issued a company laptop running either macOS or Linux, based on preference. Windows machines are available upon special request. The company uses Google Workspace for email and collaboration, Slack for messaging, and Jira for project management.

Two-factor authentication (2FA) is mandatory for all company accounts. Passwords must be at least 16 characters long and rotated every 90 days. The IT security team conducts quarterly penetration testing and annual SOC 2 Type II audits. NovaTech achieved ISO 27001 certification in 2023.

All source code is stored in GitHub Enterprise. The company follows a trunk-based development workflow with mandatory code reviews. Each pull request requires approval from at least two reviewers before merging.

Chapter 5: Environmental and Social Responsibility

NovaTech has committed to achieving carbon neutrality by 2028. The company currently offsets 80 percent of its carbon emissions through verified reforestation projects in Brazil and Kenya. All four offices run on 100 percent renewable energy.

The NovaTech Foundation donates 1 percent of annual revenue to STEM education programmes in underserved communities. In 2024, the foundation distributed $2.4 million in grants to 35 non-profit organisations across 12 countries.

Employees are given two paid volunteer days per year to support causes of their choice. The company also organises an annual hackathon focused on social-impact projects, with the winning team receiving $25,000 in funding to develop their prototype further.
"""


def create_pdf(output_path: str):
    """
    Create a minimal valid PDF file from the sample text.
    We use pypdf or fall back to manual PDF creation.
    """
    try:
        # Try using reportlab if available (for nicer formatting)
        from reportlab.lib.pagesizes import letter
        from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
        from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
        from reportlab.lib.units import inch

        doc = SimpleDocTemplate(output_path, pagesize=letter)
        styles = getSampleStyleSheet()
        
        # Create a custom style for body text
        body_style = ParagraphStyle(
            'CustomBody',
            parent=styles['Normal'],
            fontSize=11,
            leading=15,
            spaceAfter=8,
        )
        heading_style = ParagraphStyle(
            'CustomHeading',
            parent=styles['Heading1'],
            fontSize=16,
            spaceAfter=12,
            spaceBefore=18,
        )

        story = []
        for line in SAMPLE_TEXT.strip().split("\n"):
            line = line.strip()
            if not line:
                story.append(Spacer(1, 0.2 * inch))
            elif line.startswith("Chapter") or line.startswith("NovaTech Solutions"):
                # Escape any XML special characters for reportlab
                safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(safe, heading_style))
            else:
                safe = line.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                story.append(Paragraph(safe, body_style))

        doc.build(story)
        print(f"[OK] Created PDF with reportlab: {output_path}")

    except ImportError:
        # Fallback: use fpdf2 or create a simple text-based PDF manually
        try:
            from fpdf import FPDF

            pdf = FPDF()
            pdf.set_auto_page_break(auto=True, margin=15)
            pdf.add_page()
            pdf.set_font("Helvetica", size=11)

            for line in SAMPLE_TEXT.strip().split("\n"):
                line = line.strip()
                if not line:
                    pdf.ln(5)
                elif line.startswith("Chapter") or line.startswith("NovaTech Solutions"):
                    pdf.set_font("Helvetica", style="B", size=14)
                    pdf.multi_cell(0, 8, line)
                    pdf.set_font("Helvetica", size=11)
                    pdf.ln(3)
                else:
                    pdf.multi_cell(0, 6, line)
                    pdf.ln(2)

            pdf.output(output_path)
            print(f"[OK] Created PDF with fpdf2: {output_path}")

        except ImportError:
            # Last resort: write raw PDF bytes
            _write_raw_pdf(output_path)
            print(f"[OK] Created raw PDF: {output_path}")


def _write_raw_pdf(output_path: str):
    """
    Write a minimal valid PDF with the sample text.
    This is a fallback that creates a basic but valid PDF without any libraries.
    """
    # We'll write the text as a simple PDF with one stream object
    text_lines = SAMPLE_TEXT.strip().replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    
    # Build PDF content line by line for the text rendering
    pdf_text_commands = []
    y_position = 750  # Start from top
    
    for line in SAMPLE_TEXT.strip().split("\n"):
        line = line.strip()
        if not line:
            y_position -= 15
            continue
        
        # Escape special PDF characters
        safe_line = line.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        
        if y_position < 50:
            # We'd need a new page, but for simplicity just stop
            break
            
        if line.startswith("Chapter") or line.startswith("NovaTech Solutions"):
            pdf_text_commands.append(f"BT /F1 14 Tf 50 {y_position} Td ({safe_line}) Tj ET")
            y_position -= 20
        else:
            # Split long lines
            while len(safe_line) > 90:
                split_at = safe_line[:90].rfind(" ")
                if split_at == -1:
                    split_at = 90
                part = safe_line[:split_at]
                safe_line = safe_line[split_at:].strip()
                pdf_text_commands.append(f"BT /F1 10 Tf 50 {y_position} Td ({part}) Tj ET")
                y_position -= 14
                if y_position < 50:
                    break
            if y_position >= 50 and safe_line:
                pdf_text_commands.append(f"BT /F1 10 Tf 50 {y_position} Td ({safe_line}) Tj ET")
                y_position -= 14
    
    stream_content = "\n".join(pdf_text_commands)
    
    # Build the PDF structure
    pdf_parts = []
    offsets = []
    
    # Header
    pdf_parts.append(b"%PDF-1.4\n")
    
    # Object 1: Catalog
    offsets.append(len(b"".join(pdf_parts)))
    pdf_parts.append(b"1 0 obj\n<< /Type /Catalog /Pages 2 0 R >>\nendobj\n")
    
    # Object 2: Pages
    offsets.append(len(b"".join(pdf_parts)))
    pdf_parts.append(b"2 0 obj\n<< /Type /Pages /Kids [3 0 R] /Count 1 >>\nendobj\n")
    
    # Object 3: Page
    offsets.append(len(b"".join(pdf_parts)))
    pdf_parts.append(b"3 0 obj\n<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>\nendobj\n")
    
    # Object 4: Content stream
    stream_bytes = stream_content.encode("latin-1")
    offsets.append(len(b"".join(pdf_parts)))
    pdf_parts.append(f"4 0 obj\n<< /Length {len(stream_bytes)} >>\nstream\n".encode())
    pdf_parts.append(stream_bytes)
    pdf_parts.append(b"\nendstream\nendobj\n")
    
    # Object 5: Font
    offsets.append(len(b"".join(pdf_parts)))
    pdf_parts.append(b"5 0 obj\n<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>\nendobj\n")
    
    # Cross-reference table
    xref_offset = len(b"".join(pdf_parts))
    pdf_parts.append(b"xref\n")
    pdf_parts.append(f"0 {len(offsets) + 1}\n".encode())
    pdf_parts.append(b"0000000000 65535 f \n")
    for off in offsets:
        pdf_parts.append(f"{off:010d} 00000 n \n".encode())
    
    # Trailer
    pdf_parts.append(f"trailer\n<< /Size {len(offsets) + 1} /Root 1 0 R >>\n".encode())
    pdf_parts.append(f"startxref\n{xref_offset}\n%%EOF\n".encode())
    
    with open(output_path, "wb") as f:
        for part in pdf_parts:
            f.write(part)


if __name__ == "__main__":
    output = os.path.join(os.path.dirname(__file__), "sample_handbook.pdf")
    create_pdf(output)
