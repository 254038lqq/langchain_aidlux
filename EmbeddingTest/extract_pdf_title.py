import PyPDF2
import re

# 1.DocumentLoader
pdf_path = "./EmbeddingTest/Owners_Manual.pdf"
# pdf_path = "./EmbeddingTest/llm_kv.pdf"
title_pdf = []
def get_outlines_sub(outlines, level=0):
    for item in outlines:
        if type(item) == list:
            get_outlines_sub(item, level+1)
        else:
            title = item.title
            # print(title)    
            print(' ' * level + f'Level {level}: {title}')
            title_pdf.append([level,title])


# 打开PDF文件
with open(pdf_path, 'rb') as pdf_file:
    reader = PyPDF2.PdfReader(pdf_file)
    outlines = reader.outline
    get_outlines_sub(outlines)


savelabelsname_title = "./vector_fast/Owners_Manual_title.txt"
with open(savelabelsname_title,'w') as f:
    for line_list  in title_pdf:
        line = line_list[1] 
        line_0 =line.replace('©',' ') 
        line_0 =line_0.replace('◦',' ') 
        line_0 =line_0.replace('•',' ') 
        line_0 =line_0.replace('ö','o') 
        line_0 =line_0.replace('®','') 
        line_0 =line_0.replace('™','') 
        line_0 =line_0.replace('°','') 
        line_0 =line_0.replace('✓','') 
        line_0 =line_0.replace('°','') 
        f.write(str(line_list[0]) +' ')
        f.write(line_0 +'\n')


print("ok!!!!!")

###########################
# pdf_file = open(pdf_path,"rb")
pdf_reader = PyPDF2.PdfReader(pdf_path)
content= ""
for page_num in range(len(pdf_reader.pages)):
    page = pdf_reader.pages[page_num]
    text= page.extract_text()
    lines = text.split("\n")
    for line in lines:
        if line.startswith("##"):
            print(line)
# pdf_file.close()
print("ok!")


