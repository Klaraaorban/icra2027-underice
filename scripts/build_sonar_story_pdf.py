"""Short PDF: the sonar-vs-baselines story, headline result + visual proof."""
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.lib import colors
from reportlab.platypus import (SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
                                 Image, KeepTogether)

OUT = "E:/Research/Holo/icra2027_underice/results/Sonar_Story.pdf"
IMG_CLOSE = "E:/Research/Holo/gaussian-splatting/data/ps117_39_30to90_fisheye/images/f001060.jpg"
IMG_FAR = "E:/Research/Holo/gaussian-splatting/data/ps117_39_30to90_fisheye/images/f001660.jpg"

styles = getSampleStyleSheet()
title_style = ParagraphStyle("TitleX", parent=styles["Title"], fontSize=18, spaceAfter=6)
h2 = ParagraphStyle("H2", parent=styles["Heading2"], fontSize=13, spaceBefore=14, spaceAfter=6)
body = ParagraphStyle("BodyX", parent=styles["Normal"], fontSize=10.3, leading=14.5)
caption = ParagraphStyle("Caption", parent=styles["Normal"], fontSize=8.5, leading=11,
                          textColor=colors.grey, alignment=1)
small = ParagraphStyle("Small", parent=styles["Normal"], fontSize=9, leading=12.5, textColor=colors.grey)

story = []


def P(text, style=body):
    story.append(Paragraph(text, style))


def table(data, col_widths):
    t = Table(data, colWidths=col_widths, repeatRows=1)
    t.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#1f3a5f")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTSIZE", (0, 0), (-1, -1), 9.5),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.HexColor("#cccccc")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#f2f5f9")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


# ---------------------------------------------------------------- Title
story.append(Paragraph("Sonar Beats Vision: A Real Depth Check for Under-Ice 3DGS", title_style))
P("PS117-39, Antarctic under-ice ROV footage &mdash; sonar-supervised reconstruction vs. two "
  "underwater-physics baselines", small)
story.append(Spacer(1, 10))

# ---------------------------------------------------------------- 1. Setup
P("1. The setup", h2)
P("The dataset is real Antarctic ROV footage with a single-beam Valeport altimeter (sonar) logging "
  "range-to-ice at 3.92&nbsp;Hz alongside the camera. That gives an independent, physical ground truth "
  "for depth &mdash; something photometric metrics like PSNR/SSIM/LPIPS cannot provide, since they only "
  "measure how well a render matches a photo, not whether the underlying 3D geometry is correct.")
P("Our method uses this sonar trace as a training signal: a self-diagnosing depth-regularization loss "
  "that checks its own agreement with sonar and scales its own weight accordingly. The question this "
  "report answers: how much does that actually buy us, compared to reconstructing without it?")

# ---------------------------------------------------------------- 2. Method
P("2. How accuracy is measured", h2)
P("Every model is scored the same way: render each camera's depth, take the value at the image center "
  "(matching the altimeter's forward-looking beam), match it to the nearest real sonar reading by "
  "timestamp (mean gap 0.060&nbsp;s, max 0.113&nbsp;s &mdash; within one ping period at 3.92&nbsp;Hz), "
  "fit one scale factor on a training split, then measure Pearson r and RMSE on a held-out test split "
  "the scale factor never saw. All three models are evaluated with the same 5-fold contiguous "
  "along-track block holdout: adjacent frames on a slow-moving ROV are highly correlated, so a "
  "stricter, non-adjacent block is a harder and fairer test than skipping every 8th frame. "
  "Results are reported as mean&nbsp;&plusmn;&nbsp;std across the five folds.")

# ---------------------------------------------------------------- 3. Result table
P("3. Headline result", h2)
data = [
    ["Method", "PSNR", "Sonar r (5-fold CV)", "Sonar RMSE"],
    ["Ours (sonar-supervised)", "36.44 dB", "+0.980 \u00b1 0.013", "0.075 \u00b1 0.022 m"],
    ["SeaSplat (no sonar)", "21.75 dB", "\u22120.234 \u00b1 0.771", "1.460 \u00b1 0.397 m"],
    ["WaterSplatting (no sonar)", "30.24 dB", "\u22120.245 \u00b1 0.754", "1.416 \u00b1 0.387 m"],
]
story.append(table(data, col_widths=[2.2 * inch, 1.1 * inch, 1.8 * inch, 1.6 * inch]))
story.append(Spacer(1, 10))
P("Both baselines show train-split correlation of r&nbsp;&asymp;&nbsp;&minus;0.85 consistently across "
  "all five folds &mdash; as the ROV's real distance from the ice increases, their rendered depth "
  "systematically <i>decreases</i>. The scale fitted on that sign-inverted training data is spatially "
  "inconsistent, so test-block r varies widely (including flipping positive in some blocks), giving "
  "high variance in the final numbers. Two independent codebases (different SfM initialisation, "
  "different underwater rendering models) land on the same structural failure, which is strong "
  "evidence this is a gap in vision-only underwater reconstruction, not a fluke of one implementation. "
  "Neither baseline has any mechanism to anchor absolute, or even correctly-signed, depth without an "
  "external reference &mdash; exactly the gap a cheap sonar closes.")
P("Our method also has the highest PSNR of the three, despite carrying the extra sonar-loss term "
  "&mdash; geometric accuracy is not traded away for photometric quality here.")

story.append(Spacer(1, 4))

# ---------------------------------------------------------------- 4. Visual proof
P("4. Is the sonar reading actually right? A direct visual check", h2)
P("Before trusting a correlation number, it's worth looking at the actual frames. The closest and "
  "farthest sonar readings in the sequence should look close and far in the photos too.")

img_w = 2.55 * inch
img_h = img_w * (1139 / 2229)
img_row = Table(
    [[Image(IMG_CLOSE, width=img_w, height=img_h), Image(IMG_FAR, width=img_w, height=img_h)],
     [Paragraph("Closest reading: sonar = 0.105 m (frame f001060)", caption),
      Paragraph("Farthest reading: sonar = 2.925 m (frame f001660)", caption)]],
    colWidths=[img_w + 4, img_w + 4],
)
img_row.setStyle(TableStyle([("ALIGN", (0, 0), (-1, -1), "CENTER"), ("TOPPADDING", (0, 1), (-1, 1), 4)]))
story.append(KeepTogether(img_row))
story.append(Spacer(1, 8))
P("The close frame shows sharp, large-scale ice structure filling the whole image &mdash; consistent "
  "with being 10&nbsp;cm away. The far frame is visibly hazier and lower-contrast, consistent with more "
  "water-column scattering at nearly 3&nbsp;m range. The sonar trace and its frame-matching are real; "
  "the sign-inversion measured above is a genuine property of the baselines' reconstructed geometry, "
  "not a bug in how the ground truth was read.")

# ---------------------------------------------------------------- 5. Why this happens
P("5. Why vision-only reconstruction gets this backwards", h2)
P("This project independently found the same failure mode on a harder station (PS117-29): an initially "
  "reasonable point cloud (from a feed-forward SfM method, right-signed, r=+0.067) was actively "
  "<i>corrupted</i> by unconstrained photometric training alone (r dropped to &minus;0.31), because on "
  "low-texture ice, many different depth configurations explain the photos equally well. SeaSplat and "
  "WaterSplatting have no sonar and no depth regularization at all, so nothing stops their optimizers "
  "from settling on a photo-consistent but geometrically backwards solution. A cheap single-beam "
  "altimeter, most ROVs already carry one, is exactly the missing anchor.")

# ---------------------------------------------------------------- 6. Bottom line
P("6. Bottom line", h2)
P("This is not \"our sonar trick gives a small edge.\" It's that state-of-the-art underwater 3DGS/NeRF "
  "reconstruction gets the <i>direction</i> of depth wrong on real polar footage &mdash; independently, "
  "twice &mdash; and a sensor that costs a fraction of the camera rig fixes it, while also improving "
  "photometric quality. Reproduction code for exactly the numbers in Section&nbsp;3 (three short "
  "evaluation scripts, no retraining required) accompanies this report.")

doc = SimpleDocTemplate(OUT, pagesize=letter,
                         topMargin=0.7 * inch, bottomMargin=0.7 * inch,
                         leftMargin=0.75 * inch, rightMargin=0.75 * inch)
doc.build(story)
print(f"wrote {OUT}")
