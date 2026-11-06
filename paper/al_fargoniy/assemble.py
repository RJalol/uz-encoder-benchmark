# -*- coding: utf-8 -*-
import copy, re
from docx import Document
from docx.shared import Pt, Cm
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
from docx.oxml import OxmlElement
import front as F
d=Document('base.docx')
TNR="Times New Roman"
def setfont(style,size,bold=None,italic=None):
    f=style.font; f.name=TNR; f.size=Pt(size)
    if bold is not None: f.bold=bold
    if italic is not None: f.italic=italic
    rpr=style.element.get_or_add_rPr(); rf=rpr.find(qn('w:rFonts'))
    if rf is None: rf=OxmlElement('w:rFonts'); rpr.append(rf)
    for a in ('w:ascii','w:hAnsi','w:cs','w:eastAsia'): rf.set(qn(a),TNR)
    for a in ('w:asciiTheme','w:hAnsiTheme','w:cstheme','w:eastAsiaTheme'):
        if rf.get(qn(a)) is not None: del rf.attrib[qn(a)]
for st in d.styles:
    try:
        if st.type==1: setfont(st,14) if st.name not in ('Heading 1','Heading 2') else None
    except Exception: pass
for nm in ('Normal','Body Text','First Paragraph','Compact'):
    if nm in [s.name for s in d.styles]:
        st=[x for x in d.styles if x.name==nm][0]; setfont(st,14)
        pf=st.paragraph_format; pf.space_before=Pt(0); pf.space_after=Pt(0); pf.line_spacing=1.0
        pf.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
        if nm!='Compact': pf.first_line_indent=Cm(1.25)
for nm,al in (('Heading 1',WD_ALIGN_PARAGRAPH.LEFT),('Heading 2',WD_ALIGN_PARAGRAPH.LEFT)):
    st=[x for x in d.styles if x.name==nm][0]; setfont(st,14,bold=True,italic=False)
    st.font.color.rgb=None
    from docx.shared import RGBColor
    st.font.color.rgb=RGBColor(0,0,0)
    pf=st.paragraph_format; pf.space_before=Pt(6); pf.space_after=Pt(0); pf.keep_with_next=True; pf.alignment=al; pf.first_line_indent=Cm(1.25)
sec=d.sections[0]
sec.page_width=Cm(21.0); sec.page_height=Cm(29.7)
sec.left_margin=Cm(3); sec.right_margin=Cm(1.5); sec.top_margin=Cm(2); sec.bottom_margin=Cm(2)
sec.header_distance=Cm(1.0); sec.footer_distance=Cm(1.0)
# page numbers in footer (centered)
fp=sec.footer.paragraphs[0] if sec.footer.paragraphs else sec.footer.add_paragraph()
fp.alignment=WD_ALIGN_PARAGRAPH.CENTER
r=fp.add_run(); 
for t,txt in (('begin',None),('instr',' PAGE '),('end',None)):
    if t=='instr':
        e=OxmlElement('w:instrText'); e.set(qn('xml:space'),'preserve'); e.text=txt
    else:
        e=OxmlElement('w:fldChar'); e.set(qn('w:fldCharType'),t)
    r._r.append(e)
r.font.name=TNR; r.font.size=Pt(12)
body=d.element.body
paras=d.paragraphs
# --- style paragraphs: captions, tables, references, decl
for p in paras:
    t=p.text.strip()
    if re.match(r'^Table \d+\.',t):
        p.alignment=WD_ALIGN_PARAGRAPH.CENTER; p.paragraph_format.first_line_indent=Cm(0)
        p.paragraph_format.keep_with_next=True; p.paragraph_format.space_before=Pt(6)
        for r in p.runs: r.font.size=Pt(12)
    elif re.match(r'^\[\d+\] ',t):
        p.paragraph_format.first_line_indent=Cm(0); p.alignment=WD_ALIGN_PARAGRAPH.JUSTIFY
        for r in p.runs: r.font.size=Pt(12)
    elif re.match(r'^\(\d\) ',t):
        p.paragraph_format.first_line_indent=Cm(0); p.paragraph_format.left_indent=Cm(1.25)
for tb in d.tables:
    tb.alignment=1
    tblPr=tb._tbl.tblPr
    for old in tblPr.findall(qn('w:tblBorders')): tblPr.remove(old)
    bd=OxmlElement('w:tblBorders')
    for k in ('top','left','bottom','right','insideH','insideV'):
        e=OxmlElement('w:'+k); e.set(qn('w:val'),'single'); e.set(qn('w:sz'),'4'); e.set(qn('w:space'),'0'); e.set(qn('w:color'),'000000'); bd.append(e)
    tblPr.append(bd)
    ncol=len(tb.columns)
    mx=[max(len(r.cells[j].text) for r in tb.rows) for j in range(ncol)]
    mx=[max(m,7) for m in mx]
    tot=sum(mx); W=16.3
    ws=[W*m/tot for m in mx]
    for old in tblPr.findall(qn('w:tblLayout')): tblPr.remove(old)
    lay=OxmlElement('w:tblLayout'); lay.set(qn('w:type'),'fixed'); tblPr.append(lay)
    for old in tblPr.findall(qn('w:tblW')): tblPr.remove(old)
    tw=OxmlElement('w:tblW'); tw.set(qn('w:w'),str(int(sum(ws)/2.54*1440))); tw.set(qn('w:type'),'dxa'); tblPr.append(tw)
    grid=tb._tbl.tblGrid
    for j,gc in enumerate(grid.findall(qn('w:gridCol'))): gc.set(qn('w:w'),str(int(ws[j]/2.54*1440)))
    for row in tb.rows:
        for j,c in enumerate(row.cells): c.width=Cm(ws[j])
    for ri,row in enumerate(tb.rows):
        for c in row.cells:
            for p in c.paragraphs:
                p.paragraph_format.first_line_indent=Cm(0); p.alignment=WD_ALIGN_PARAGRAPH.CENTER
                p.paragraph_format.space_after=Pt(0)
                for r in p.runs:
                    r.font.size=Pt(11); r.font.name=TNR
                    if ri==0: r.font.bold=True
for tb in d.tables:
    rows=tb.rows
    for ri,row in enumerate(rows):
        trPr=row._tr.get_or_add_trPr(); cs=OxmlElement('w:cantSplit'); trPr.append(cs)
        if ri<len(rows)-1:
            for c in row.cells:
                for p in c.paragraphs: p.paragraph_format.keep_with_next=True
    nxt=tb._tbl.getnext()
    if nxt is not None and nxt.tag==qn('w:p'):
        from docx.text.paragraph import Paragraph
        Paragraph(nxt,d).paragraph_format.space_before=Pt(6)
# --- front matter inserted before first paragraph
first=d.paragraphs[0]
def addp(text,bold=False,align=WD_ALIGN_PARAGRAPH.CENTER,size=14,indent=0,upper=False,label=None,italic=False,sb=0):
    p=first.insert_paragraph_before()
    p.style=[x for x in d.styles if x.name=='Normal'][0]
    p.alignment=align; p.paragraph_format.first_line_indent=Cm(indent)
    p.paragraph_format.space_before=Pt(sb); p.paragraph_format.space_after=Pt(0)
    if label:
        r=p.add_run(label+" "); r.bold=True; r.font.size=Pt(size); r.font.name=TNR
    r=p.add_run(text.upper() if upper else text); r.bold=bold; r.italic=italic; r.font.size=Pt(size); r.font.name=TNR
    return p
for lang,title,auth,abs_,kw in (('uz',F.UZ_TITLE,F.UZ_AUTH,F.UZ_ABS,F.UZ_KW),('en',F.EN_TITLE,F.EN_AUTH,F.EN_ABS,F.EN_KW),('ru',F.RU_TITLE,F.RU_AUTH,F.RU_ABS,F.RU_KW)):
    addp(title,bold=True,upper=True,sb=6 if lang!='uz' else 0)
    addp("",size=14)
    for line in auth: addp(line,align=WD_ALIGN_PARAGRAPH.RIGHT)
    addp("Email: "+F.EMAIL,align=WD_ALIGN_PARAGRAPH.RIGHT)
    addp("ORCID: "+F.ORCID,align=WD_ALIGN_PARAGRAPH.RIGHT)
    addp("",size=14)
    addp(abs_,align=WD_ALIGN_PARAGRAPH.JUSTIFY,indent=1.25,label=LB[0] if False else F.LBL[lang][0])
    addp(kw,align=WD_ALIGN_PARAGRAPH.JUSTIFY,indent=1.25,label=F.LBL[lang][1])
    addp("",size=14)
d.core_properties.title=F.EN_TITLE; d.core_properties.author="Jaloliddin Rajabov"
d.save('out.docx')
