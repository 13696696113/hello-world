import matplotlib.pyplot as plt
import xlrd
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import rc
import xlwt

def read_excel(input_file_name):
    """
    从xls文件中读取数据
    """
    workbook = xlrd.open_workbook(input_file_name)
    # 通过sheet_by_name()方法获取到一张表，返回一个对象
    table = workbook.sheet_by_name(workbook.sheet_names()[0])
    # 通过nrows和ncols获取到表格中数据的行数和列数
    rows = table.nrows
    # cols = table.ncols
    result = []
    # 可以通过row.values()按行获取数据，返回一个列表，也可以按列
    for row in range(rows):
        row_data = table.row_values(row)
        result.append(row_data)

    return result


def to_excel(timecost_pcde_h):
    '''
    timecost_pcde_h存进execl
    '''
    matrix = []
    for i in range(14):
        temp = []
        temp.append(i+6)
        temp += timecost_pcde_h[i*30:(i*30)+30]
        matrix.append(temp)
    # 数据放进excel
    workbook = xlwt.Workbook(encoding='utf-8')  # 设置一个workbook，其编码是utf-8
    worksheet = workbook.add_sheet("test_sheet")
    for i in range(len(matrix)):  # 循环将a和b列表的数据插入至excel
        for j in range(len(matrix[0])):
            worksheet.write(j, i, label=matrix[i][j])
    workbook.save(r"C:\Users\22174\Desktop\区块链+物联网\实验代码\加密成本\混合加密_本机解密数据.xls")


def PCdouble_to_excel():
    '''
        调整格式
    '''
    x = read_excel(r"C:\Users\22174\Desktop\区块链+物联网\实验代码\加密成本\对称加密_本机解密数据.xls")
    y = read_excel(r"C:\Users\22174\Desktop\区块链+物联网\实验代码\加密成本\混合加密_本机解密数据.xls")
    matrix = []
    for i in range(len(x)):
        for j in range(1, len(x[0])):
            temp = []
            temp.append(x[i][j])
            temp.append(i + 6)
            temp.append("Symmetric decryption by PC")
            matrix.append(temp)
    for i in range(len(y[0])):
        for j in range(1, len(y)):
            if y[j][i] > 1:
                temp = []
                temp.append(y[j][i])
                temp.append(i + 6)
                temp.append("Hybrid decryption by PC")
                matrix.append(temp)
    workbook = xlwt.Workbook(encoding='utf-8')  # 设置一个workbook，其编码是utf-8
    worksheet = workbook.add_sheet("test_sheet")
    worksheet.write(0, 0, label="value")
    worksheet.write(0, 1, label="length")
    worksheet.write(0, 2, label="kind")
    for i in range(len(matrix)):  # 循环将a和b列表的数据插入至excel
        for j in range(len(matrix[0])):
            worksheet.write(i+1, j, label=matrix[i][j])
    workbook.save(r"C:\Users\22174\Desktop\区块链+物联网\实验代码\加密成本\test.xls")


def RASPdouble_to_excel():
    x = read_excel(r"C:\Users\22174\Desktop\区块链+物联网\实验代码\加密成本\对称加密_树莓派加密数据.xls")
    y = read_excel(r"C:\Users\22174\Desktop\区块链+物联网\实验代码\加密成本\混合加密_树莓派加密数据.xls")
    matrix = []
    for i in range(len(x)):
        for j in range(1, len(x[0])):
            temp = []
            temp.append(x[i][j])
            temp.append(i + 6)
            temp.append("Symmetric encryption by Raspberry Pi")
            matrix.append(temp)
    for i in range(len(y[0])):
        for j in range(1, len(y)):
            temp = []
            temp.append(y[j][i])
            temp.append(i + 6)
            temp.append("Hybrid encryption by Raspberry Pi")
            matrix.append(temp)
    workbook = xlwt.Workbook(encoding='utf-8')  # 设置一个workbook，其编码是utf-8
    worksheet = workbook.add_sheet("test_sheet")
    worksheet.write(0, 0, label="value")
    worksheet.write(0, 1, label="length")
    worksheet.write(0, 2, label="kind")
    for i in range(len(matrix)):  # 循环将a和b列表的数据插入至excel
        for j in range(len(matrix[0])):
            worksheet.write(i + 1, j, label=matrix[i][j])
    workbook.save(r"C:\Users\22174\Desktop\区块链+物联网\实验代码\加密成本\test.xls")


def drawDecrypt():
    plt.style.use("seaborn-white")
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 放大
    plt.figure(figsize=(10, 8))
    # 坐标轴范围
    plt.xlim((0, 20))
    plt.ylim((-1, 20))
    bwith = 1.1  # 边框宽度设置为1.1
    # 图边框设置
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    tips = pd.read_csv('PC两种解密耗时画图.csv')
    tips.head()
    sns.boxplot(x="length", y="value", hue="kind", data=tips, notch=True, whis=6, fliersize=3, palette="Set3")
    plt.plot(length0, timecost_pcde, color="blue", linewidth=2,
             label='Average value of symmetric decryption by PC')  # 均值折线图
    plt.plot(length0, timecost_pcde_h1, color="red", linewidth=2, label='Average value of hybrid decryption by PC')

    plt.grid()
    plt.tick_params(labelsize=12)  # 坐标轴刻度字体大小
    plt.xlabel('Message length ($\\rm log_{2}bytes$)',
               fontdict={'family': 'Times New Roman', 'weight': 700, 'size': 13})
    plt.ylabel('Time consumption (ms)', fontdict={'family': 'Times New Roman', 'weight': 700, 'size': 13})
    plt.legend(loc='best', frameon=True, edgecolor='grey',
               prop={'family': 'Times New Roman', "size": 12, "weight": "black"})

    plt.savefig("D:\桌面文件\区块链+物联网\SPE投稿\二审修改\图片更新\decode.eps", dpi=1000, format='eps')
    plt.show()


def drawEncrypt():
    plt.style.use("seaborn-white")
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.sans-serif'] = ['SimHei']
    # 放大
    plt.figure(figsize=(10, 8))
    # 坐标轴范围
    plt.xlim((0, 20))
    # plt.ylim((-1, 19))
    bwith = 1.1  # 边框宽度设置为1.1
    # 图边框设置
    ax = plt.gca()  # 获取边框
    ax.spines['bottom'].set_linewidth(bwith)
    ax.spines['left'].set_linewidth(bwith)
    ax.spines['top'].set_linewidth(bwith)
    ax.spines['right'].set_linewidth(bwith)

    tips = pd.read_csv('树莓派两种加密耗时画图.csv')
    tips.head()
    print(tips)
    sns.boxplot(x="length", y="value", hue="kind", data=tips, notch=True, whis=3, fliersize=3, palette="Set3")
    print(length0)
    plt.plot(length0, timecost_raspen, color="blue", linewidth=2,
             label='Average value of symmetric encryption by Raspberry Pi')  # 均值折线图
    plt.plot(length0, timecost_raspen_h, color="red", linewidth=2,
             label='Average value of hybrid encryption by Raspberry Pi')

    plt.grid()
    plt.tick_params(labelsize=12)  # 坐标轴刻度字体大小
    plt.xlabel('Message length ($\\rm log_{2}bytes$)',
               fontdict={'family': 'Times New Roman', 'weight': 700, 'size': 13})  # 'normal'
    plt.ylabel('Time consumption (ms)', fontdict={'family': 'Times New Roman', 'weight': 700, 'size': 13})
    plt.legend(loc='best', frameon=True, edgecolor='grey',
               prop={'family': 'Times New Roman', "size": 12, "weight": "black"})

    plt.savefig("D:\桌面文件\区块链+物联网\SPE投稿\二审修改\图片更新\encode.eps", dpi=1000, format='eps')
    plt.show()


if __name__ == '__main__':
    length = [6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19]
    length0 = []
    for i in range(len(length)):
        length0.append(length[i]-6)

    # 对称加密——本机解密(平均值)
    timecost_pcde = [0.01808929443359375, 0.010944843292236328, 0.015041351318359375, 0.014065742492675781, 0.02691173553466797, 0.0340428352355957, 0.050884246826171875, 0.09707522392272949, 0.16710329055786133, 0.34076452255249023, 0.5820167064666748, 1.1239633560180664, 2.274588108062744, 4.959190368652344]

    # 混合加密——本机解密
    timecost_pcde_h = [8.07046890258789, 8.067846298217773, 9.06682014465332, 8.002042770385742, 5.004644393920898,
                 5.959749221801758, 9.067296981811523, 9.024620056152344, 9.067058563232422, 9.011507034301758,
                 8.962154388427734, 10.03265380859375, 9.000778198242188, 6.006479263305664, 4.970073699951172,
                 6.000041961669922, 4.999876022338867, 7.012844085693359, 7.006645202636719, 8.998870849609375,
                 5.000829696655273, 6.003141403198242, 4.978418350219727, 6.009817123413086, 0.0, 8.999109268188477,
                 6.004571914672852, 15.629053115844727, 4.9610137939453125, 5.0048828125, 5.965232849121094,
                 6.007909774780273, 4.999637603759766, 4.996061325073242, 8.013486862182617, 6.006002426147461,
                 5.003213882446289, 4.962682723999023, 7.002353668212891, 10.062217712402344, 10.073661804199219,
                 9.021520614624023, 6.054162979125977, 10.06627082824707, 6.050825119018555, 9.070873260498047,
                 7.983922958374023, 6.876945495605469, 9.066343307495117, 12.067317962646484, 10.067224502563477,
                 9.053468704223633, 10.075569152832031, 8.027791976928711, 9.01031494140625, 10.037422180175781,
                 10.034561157226562, 8.953571319580078, 14.967918395996094, 10.066032409667969, 0.0, 11.992692947387695,
                 8.45026969909668, 8.063316345214844, 9.070158004760742, 8.962631225585938, 7.967710494995117,
                 9.06825065612793, 6.008386611938477, 7.973432540893555, 9.020566940307617, 7.014274597167969,
                 6.984710693359375, 9.065389633178711, 9.071111679077148, 10.031461715698242, 9.067535400390625,
                 9.031057357788086, 10.031700134277344, 5.001306533813477, 5.002260208129883, 9.057760238647461,
                 10.066747665405273, 10.067462921142578, 10.042428970336914, 8.070707321166992, 8.069038391113281,
                 9.068012237548828, 9.986639022827148, 10.032892227172852, 9.925127029418945, 7.975101470947266,
                 9.020090103149414, 10.059595108032227, 9.07135009765625, 5.976438522338867, 10.030746459960938,
                 11.477470397949219, 6.913423538208008, 9.011507034301758, 11.987447738647461, 10.042428970336914,
                 12.025117874145508, 10.071754455566406, 9.071826934814453, 9.017467498779297, 9.057283401489258,
                 8.068323135375977, 9.02414321899414, 10.066032409667969, 10.070085525512695, 10.065317153930664,
                 8.072137832641602, 7.982492446899414, 4.96673583984375, 8.05354118347168, 9.018421173095703,
                 7.026910781860352, 10.07223129272461, 9.074211120605469, 9.070158004760742, 9.062767028808594,
                 8.06879997253418, 9.064674377441406, 8.95237922668457, 9.073972702026367, 10.032176971435547, 0.0,
                 7.962465286254883, 8.966207504272461, 8.925199508666992, 9.059906005859375, 10.071754455566406,
                 15.316009521484375, 5.002260208129883, 8.030176162719727, 12.06517219543457, 9.019613265991211,
                 8.050918579101562, 10.068655014038086, 10.067224502563477, 12.065410614013672, 8.071422576904297,
                 9.91368293762207, 10.068178176879883, 7.985830307006836, 9.06991958618164, 11.032581329345703,
                 9.066104888916016, 7.939338684082031, 12.037515640258789, 9.07135009765625, 12.076139450073242,
                 7.049083709716797, 8.033037185668945, 7.067680358886719, 9.00888442993164, 7.975578308105469,
                 10.062694549560547, 10.084152221679688, 10.065555572509766, 9.07278060913086, 7.985830307006836,
                 10.071516036987305, 10.08296012878418, 10.064840316772461, 7.989645004272461, 7.989645004272461,
                 8.027791976928711, 9.064197540283203, 7.9193115234375, 9.060144424438477, 7.016658782958984,
                 10.070323944091797, 9.061336517333984, 13.028383255004883, 10.040044784545898, 8.969783782958984,
                 8.98885726928711, 10.071754455566406, 9.067535400390625, 9.073019027709961, 10.069131851196289,
                 10.063886642456055, 9.03177261352539, 6.918430328369141, 10.077476501464844, 10.024547576904297,
                 5.998373031616211, 7.982969284057617, 10.027170181274414, 10.026931762695312, 10.065317153930664,
                 7.021188735961914, 5.029916763305664, 6.734132766723633, 8.988142013549805, 10.032176971435547,
                 12.068748474121094, 7.964372634887695, 11.023759841918945, 9.002923965454102, 9.050130844116211,
                 9.050130844116211, 9.01937484741211, 10.073184967041016, 8.052825927734375, 8.013486862182617,
                 9.017229080200195, 11.062383651733398, 7.066249847412109, 8.985519409179688, 8.070945739746094,
                 7.0590972900390625, 10.031700134277344, 15.546560287475586, 9.022951126098633, 8.066177368164062,
                 9.032487869262695, 9.011268615722656, 9.029388427734375, 8.97836685180664, 4.976511001586914,
                 9.0789794921875, 8.965730667114258, 9.069204330444336, 9.022712707519531, 7.919073104858398,
                 7.991552352905273, 10.060787200927734, 10.033369064331055, 8.073091506958008, 10.061264038085938,
                 9.030818939208984, 8.98432731628418, 9.054422378540039, 10.068178176879883, 5.003690719604492,
                 9.067773818969727, 8.997201919555664, 9.86933708190918, 9.04989242553711, 9.073495864868164,
                 9.032249450683594, 9.032487869262695, 10.052919387817383, 10.023355484008789, 6.988048553466797,
                 9.063005447387695, 7.063150405883789, 8.065223693847656, 8.990287780761719, 7.921457290649414,
                 10.074377059936523, 8.970022201538086, 8.066654205322266, 8.921146392822266, 9.06991958618164,
                 15.294551849365234, 10.066747665405273, 6.05463981628418, 10.067224502563477, 7.983922958374023,
                 8.065462112426758, 6.986856460571289, 8.033037185668945, 9.02247428894043, 9.068489074707031,
                 6.046056747436523, 8.993387222290039, 7.970094680786133, 8.91876220703125, 8.924245834350586,
                 6.984949111938477, 8.058786392211914, 9.066343307495117, 9.055376052856445, 7.9898834228515625,
                 7.985353469848633, 9.033203125, 8.857011795043945, 0.0, 8.980274200439453, 8.988142013549805,
                 7.926702499389648, 9.069204330444336, 9.016752243041992, 8.074045181274414, 8.057117462158203,
                 5.005598068237305, 8.052587509155273, 8.0108642578125, 8.966684341430664, 10.033369064331055,
                 9.061336517333984, 10.06627082824707, 9.92131233215332, 8.017778396606445, 9.015321731567383,
                 7.757902145385742, 9.066104888916016, 7.925510406494141, 5.757808685302734, 9.070158004760742,
                 9.017467498779297, 10.068893432617188, 12.022256851196289, 9.925603866577148, 9.068012237548828,
                 8.04758071899414, 14.031410217285156, 10.066747665405273, 9.066581726074219, 8.008480072021484,
                 9.040117263793945, 9.065866470336914, 9.06515121459961, 8.026599884033203, 9.080648422241211,
                 6.001710891723633, 6.048917770385742, 10.070085525512695, 9.027719497680664, 9.067058563232422,
                 8.97359848022461, 10.075807571411133, 9.022951126098633, 9.070634841918945, 9.067773818969727,
                 7.604122161865234, 8.005619049072266, 8.925199508666992, 8.049249649047852, 9.073019027709961,
                 9.070158004760742, 10.027408599853516, 10.071754455566406, 7.998466491699219, 8.054256439208984,
                 8.930444717407227, 7.828950881958008, 15.276432037353516, 5.925416946411133, 10.072708129882812,
                 9.546518325805664, 10.070562362670898, 9.030342102050781, 5.011796951293945, 9.93204116821289,
                 10.06937026977539, 9.08207893371582, 9.041786193847656, 9.076595306396484, 8.031606674194336,
                 8.974552154541016, 9.068489074707031, 8.11624526977539, 9.072065353393555, 8.062362670898438,
                 10.072946548461914, 5.007505416870117, 10.066509246826172, 4.955530166625977, 9.169340133666992,
                 10.935544967651367, 9.072542190551758, 14.072895050048828, 17.071008682250977, 9.072542190551758,
                 9.073972702026367, 10.07986068725586, 9.067058563232422, 9.025096893310547, 9.01937484741211,
                 10.074853897094727, 9.911537170410156, 8.072853088378906, 8.071422576904297, 7.979154586791992,
                 9.06682014465332, 9.077072143554688, 10.030984878540039, 10.069131851196289, 7.015466690063477,
                 9.064674377441406, 9.062051773071289, 9.021282196044922, 12.06064224243164, 8.909940719604492,
                 7.959842681884766, 6.71696662902832, 15.92707633972168, 10.06937026977539, 9.070396423339844,
                 10.034322738647461, 10.04791259765625, 10.031700134277344, 9.002685546875, 10.061264038085938,
                 7.95292854309082, 10.09368896484375, 9.067535400390625, 5.954265594482422, 7.0629119873046875,
                 9.026288986206055, 9.064674377441406, 9.063243865966797, 7.735013961791992, 12.028694152832031,
                 9.018182754516602, 8.070230484008789, 9.018659591674805, 9.066343307495117, 10.069608688354492,
                 8.056163787841797, 9.024620056152344, 7.963657379150391, 9.025812149047852, 17.071962356567383,
                 9.071588516235352, 7.070064544677734, 10.07223129272461, 7.958650588989258, 5.008220672607422,
                 7.933855056762695, 8.025407791137695, 9.067058563232422, 9.067296981811523, 10.063886642456055,
                 9.038686752319336, 8.063316345214844, 9.066581726074219, 12.925148010253906, 8.92329216003418,
                 7.987260818481445, 10.071277618408203, 9.029150009155273, 10.068416595458984, 8.018732070922852,
                 10.053157806396484, 8.072376251220703, 10.072946548461914, 9.070873260498047, 9.029388427734375,
                 8.073091506958008, 9.069204330444336, 5.050897598266602, 9.070396423339844, 4.999637603759766,
                 9.079217910766602, 7.035255432128906, 9.075403213500977, 9.012699127197266, 9.065866470336914,
                 10.068893432617188, 8.927583694458008, 5.979299545288086, 10.069847106933594, 7.962942123413086,
                 10.016918182373047, 7.9555511474609375, 15.310525894165039, 9.062528610229492, 9.070873260498047,
                 9.076356887817383, 7.93004035949707, 9.033441543579102, 9.069681167602539, 8.066177368164062,
                 9.026288986206055, 9.029150009155273, 8.986234664916992, 5.753517150878906, 5.8441162109375,
                 8.073091506958008, 9.070634841918945, 5.973339080810547, 10.031938552856445, 4.991292953491211,
                 6.974935531616211, 9.014368057250977, 9.072303771972656, 9.077310562133789, 8.0718994140625,
                 6.054162979125977, 9.07444953918457, 8.787870407104492, 9.032487869262695, 11.063098907470703,
                 9.010076522827148, 9.069681167602539, 9.0789794921875, 8.950233459472656, 9.066104888916016,
                 10.063886642456055, 12.080907821655273, 9.05919075012207, 9.071111679077148, 8.964776992797852,
                 9.004592895507812]
    # 将timecost_pcde_h存入excel
    # to_excel(timecost_pcde_h)
    # 调整格式
    # PCdouble_to_excel()

    # 求取混合加密——本机解密的平均值
    timecost_pcde_h1 = []
    index = 80
    while index < 500:
        tmp = timecost_pcde_h[index:index + 30]
        timecost_pcde_h1.append(np.mean(tmp))
        index += 30
    # print(timecost_pcde_h1)

    # 对称加密——树莓派加密(平均值)
    timecost_raspen = []
    x = read_excel(r"D:\桌面文件\区块链+物联网\实验代码\加密成本\对称加密_树莓派加密数据.xls")
    for i in range(len(x)):
        timecost_raspen.append(np.mean(x[i]))
    # print(timecost_raspen)

    # 混合加密——树莓派加密(平均值)
    timecost_raspen_h = [4.901055868733364, 2.850039707428219, 4.115410081703687, 4.967391796015175, 3.164534102513196, 5.475148988537335, 4.689818057731403, 4.093499035514579, 6.342203242054758, 6.7604296030690385, 10.024868854819074, 15.866133492620381, 31.773119707021575, 61.36680025856005]
    # 调整格式
    # RASPdouble_to_excel()

    # drawEncrypt()
    drawDecrypt()




# import json
# from pylab import *
# from matplotlib.font_manager import FontProperties
# # coding=utf-8
# from matplotlib import pyplot as plt
# from matplotlib import font_manager
#
# # 绘图
# # 设置X,Y的范围
# old_x = []
# for i in range(1, 161):
#     old_x.append(i)
# old_y_1 = [-0.385937604680461, -1.3508697194171244, -0.041286531160660056, -0.14824289405911117, -0.3557986543856715,
#            -0.18674142638509106, -0.34544676066111424, -0.19980796370351595, -0.5501343954502973, -0.17415794413683194,
#            -0.3268268434462023, -0.15610613837363618, -0.2520052555948078, -0.17300481508655952, -0.2554990203269687,
#            -0.26399120994839653, -0.13974443978508488, 0.10690999152072311, -0.006341558840011752,
#            -0.022499636161943393, 0.06281764638460385, 0.1535200279599216, 0.09121691822952915, 0.14855779199716734,
#            0.24561560500487734, 0.11362947746426533, 0.19597834086045862, 0.26109849788883976, 0.2661383422280492,
#            0.2557410121331771, 0.22638506975739892, 0.2735213633827407, 0.2719347303840146, 0.24091814031663183,
#            0.29320809836944195, 0.2771241542870737, 0.28170024961993145, 0.28452892964488097, 0.2820221878168143,
#            0.17020620754037663, 0.2452437400585663, 0.1549722319663075, 0.26864194593670765, 0.3025443178688706,
#            0.28988184287855745, 0.30597763925860466, 0.28876066026888214, 0.30399509395955626, 0.1425106552535076,
#            0.252555748543584, 0.30654419462211213, 0.33472087446190435, 0.31311697186219134, 0.331861560298395,
#            0.3095792431926002, 0.31935484316535423, 0.3070717281437575, 0.2875380448277707, 0.3143200530872815,
#            0.3469080103657701, 0.3269778006911124, 0.3153783157973833, 0.35169320643018864, 0.35305272512008834,
#            0.3648230289594493, 0.36137207807492555, 0.37329061130298746, 0.3712732040838932, 0.36884143269297787,
#            0.37126733660759226, 0.38233646570639235, 0.39541477737233965, 0.39579181075703274, 0.3619690204262642,
#            0.4067306438285898, 0.41029373245925427, 0.411737526335059, 0.38403555774358866, 0.4171323641246548,
#            0.42235965698212274, 0.4173083925059302, 0.4061203365705526, 0.4035137360270884, 0.4001084795198252,
#            0.4118374709477953, 0.42192536950839565, 0.4311995623171444, 0.42597581923074834, 0.4388338837729868,
#            0.4115390032932644, 0.3945044435152989, 0.4406569530692783, 0.4230643826669136, 0.45809531312149265,
#            0.3991766111812338, 0.4355193607222301, 0.46813420065924316, 0.4611845115011657, 0.45256200014449455,
#            0.4283027932172512, 0.42937304085677064, 0.4665216669912312, 0.45003590522568093, 0.4078408451835396,
#            0.4598551358850713, 0.45569774059040935, 0.4336754211318541, 0.4702182645769686, 0.4698769543474455,
#            0.4408263875837001, 0.45587521923210705, 0.4527152738711311, 0.4532340300439278, 0.45741666376456336,
#            0.4637501321283223, 0.45813086607115794, 0.4494205324080679, 0.44772754107882373, 0.41909671949004534,
#            0.4697424657472419, 0.46533102601805465, 0.45327228729005875, 0.4412593017715831, 0.4712309649922284,
#            0.4696434576489039, 0.4548126649130786, 0.4664510589520421, 0.4633633089812239, 0.4670072238652918,
#            0.471966878539688, 0.48334250514982957, 0.4823737212391397, 0.45703643091386614, 0.48526809749883104,
#            0.4836200549587527, 0.46906610703935103, 0.46978579237000684, 0.47158377680549923, 0.47543988177118757,
#            0.48076822449469303, 0.4601435674924905, 0.4402986937834992, 0.4792737713529952, 0.4910025452625776,
#            0.4893283676343634, 0.47189435069472896, 0.4726284271171236, 0.48373275340609134, 0.47804165994347414,
#            0.48530931890622375,
#            0.4594268010541255, 0.481029675610696, 0.4814140447088243, 0.4779541980658333, 0.4838978728489822,
#            0.47866227275366113, 0.48584641911608917, 0.48919883810339015, 0.4800267508512439, 0.48919604032317554]
# old_y_2 = [2.6216924, 4.4469943, 1.969737, 2.17206, 2.5646803, 2.244885, 2.545098, 2.2696025, 2.9322927, 2.2210817,
#            2.509876, 2.1869342, 2.3683405, 2.2189004, 2.3749495, 2.3910139, 2.155984, 1.689403, 1.9036338, 1.9341991,
#            1.7728097, 1.6012336, 1.7190888, 1.6106205, 1.4270221, 1.6766921, 1.5209179, 1.3977342, 1.3882006, 1.4078685,
#            1.4633994, 1.3742346, 1.377236, 1.4359081, 1.3369944, 1.3674194, 1.3587632, 1.3534123, 1.3581542, 1.5696696,
#            1.4277256, 1.5984864, 1.3834647, 1.3193337, 1.3432865, 1.3128392, 1.3454076, 1.3165892, 1.6220595, 1.4138938,
#            1.3117672, 1.2584672, 1.2993339, 1.2638761, 1.3060261, 1.2875344, 1.3107694, 1.34772, 1.2970582, 1.2354136,
#            1.2731143, 1.2950563, 1.2263618, 1.2237899, 1.201525, 1.2080529, 1.1855073, 1.1893234, 1.1939235, 1.1893345,
#            1.1683958, 1.1436563, 1.1429431, 1.2069237, 1.1222508, 1.1155108, 1.1127796, 1.1651818, 1.1025746, 1.0926864,
#            1.1022416, 1.1234052, 1.1283361, 1.1347775, 1.1125906, 1.0935079, 1.0759646, 1.085846, 1.0615233, 1.1131551,
#            1.1453784, 1.0580745, 1.0913533, 1.0250875, 1.1365403, 1.067793, 1.0060974, 1.0192438, 1.0355545, 1.0814441,
#            1.0794197, 1.0091479, 1.0403329, 1.1201507, 1.0217587, 1.0296229, 1.0712811, 1.0021552, 1.0028008, 1.057754,
#            1.029287, 1.0352646, 1.0342833, 1.0263712, 1.0143907, 1.0250201, 1.041497, 1.0446997, 1.0988587, 1.0030552,
#            1.0114002, 1.0342109, 1.0569352, 1.0002396, 1.0032425, 1.0312971, 1.0092814, 1.0151223, 1.0082294,
#            0.99884754, 0.97732884, 0.9791615, 1.0270905, 0.9736864, 0.976804, 1.0043347, 1.0029733, 0.99957216,
#            0.9922778, 0.9821985, 1.0212129, 1.0587523, 0.98502547, 0.9628389, 0.9660059, 0.99898463, 0.9975961,
#            0.97659063, 0.9873562, 0.9736083, 1.0225688, 0.98170394, 0.9809769, 0.98752165, 0.9762783, 0.9861822,
#            0.9725924, 0.96625096, 0.98360115, 0.96625626]
#
# y_1 = []
# y_2 = []
# x = []
# for i in range(0, 160, 10):
#     y_1.append(old_y_1[i])
#     y_2.append(old_y_2[i])
#     x.append(old_x[i])
#
# # print(y_1, y_2)
# # 设置图形大小
# plt.figure(figsize=(20, 8), dpi=500)
# # 支持中文
# mpl.rcParams['font.sans-serif'] = ['SimHei']
#
# # 绘制图像 label 设置标签 color设置颜色 linestyle 设置线条 linewidth 设置线条粗细 alpha设置透明度
# plt.plot(x, y_1, label='R2', color='red', linestyle=':', marker='.', markersize=5)
# plt.plot(x, y_2, label='Loss', color='black', linestyle='--', marker='.', markersize=5)
#
# # 设置X刻度
# # _xtick_labels = ['{}'.format(i) for i in x]
# _xtick_labels = x
# plt.xticks(x, _xtick_labels, rotation=45)
#
# plt.xlabel('epoch')
#
# # 绘制网格
# plt.grid(alpha=0.8)  # alpha调整网格透明度
#
# # 添加图例 先写label参数 再用plt.lenged()
# plt.legend(loc='upper left')  # 显示中文设置prop参数 loc='upper left'将图例移到左上方
#
# # 展示图形
# plt.show()