import fitz

def sumTOC(pdf): 
    sumt = 0   #记录有效toc数量
    tvalue = ""  #设置书签评估标识
    doc = fitz.open(pdf)
    toc = doc.get_toc()   #获取pdf目录
    for t in toc:             
        if t[1].isdigit() or t[1].isspace():    #统计目录是数字，空格 pass。            
            pass
        else:
            if len(t[1]) > 5:
                sumt += 1    #其他情况  比如 中文   则累加    情况包括：中英文，特殊字符  都累加。
            elif len(t[1]) < 5 :
                #含有中文的书签，才会标记为有效书签并统计数量
                for j in t[1] :    #便利字符串 看是否有中文
                    if u'\u4e00' <= j <= u'\u9fff':
                        sumt += 1
                        print("短长度书签条目，但含有中文")
                        break
    if sumt >0 and sumt <= 30:
        tvalue = "+T30_"
    elif sumt > 30 and sumt <= 80:
        tvalue = "+T80_"
    elif sumt > 80 and sumt <= 200:
        tvalue = "+T200_"
    elif sumt > 200 and sumt <= 500:
        tvalue = "+T500_"
    elif sumt > 500:
        tvalue = "+T999_"
    print("本书书签数量：   " +  str(sumt))
    #print(tvalue)
    doc.close()
    return tvalue


def readpdftitle2txt(pdf_path):
    doc = fitz.open(pdf_path)
    toc = doc.get_toc()   #获取pdf目录
    title_pdf= toc
    savelabelsname_title = "./vector_fast/Owners_Manual_title.txt"
    with open(savelabelsname_title,'w') as f:
        for line_list  in title_pdf:
            if line_list[1].isdigit() or line_list[1].isspace():    #统计目录是数字，空格 pass。  
                print(line_list[1])     
                print("null title")     
                continue
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
            f.write(str(line_list[0]) +'= =')  # 用于分割level 和页码
            f.write(line_0 +'= ='+ str(line_list[2])+'\n')



path=r"D:\algorithm\ai_agent\langchain_pt\EmbeddingTest\Owners_Manual.pdf"
# sumTOC(path)
readpdftitle2txt(path)
print("ok!")