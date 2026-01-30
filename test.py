# test_pipeline.py
"""
密码处理流水线测试：
    1) 预处理 (preprocess_breach)
    2) L/D/S 粗分 + 并行细分 (segment_l_d_s)
"""

from __future__ import annotations

# ---------------------------------------------------------------------
# 0. 依赖加载
# ---------------------------------------------------------------------
from pathlib import Path

from config import EXPAND_TOP_PATHS, MERGED_TOP_PATHS, TOPK_L, project_path
from training.build_english_dict_automaton import load_english_dict_automaton
from training.build_english_name_automaton import load_english_name_automaton

from training.segmenter.preprocessor import preprocess_breach
from training.segmenter.segment_l_d_s import build_en_automaton, segment_l_d_s
from training.segmenter.postprocessor import postprocess_multibranch
from training.training_manager import MAX_SELECTED_PATHS, POSTERIOR_THRESHOLD
import math


# ---------------------------------------------------------------------
# 1. 拼音表加载 —— segment_l_d_s 现在只接受“内存集合”，不再自带 I/O
# ---------------------------------------------------------------------
def load_chinese_pinyin_from_file(path: str | Path) -> set[str]:
    """从文本文件读取拼音词表并全部转为小写集合"""
    try:
        with open(path, encoding="utf-8") as f:
            return {ln.strip().lower() for ln in f if ln.strip()}
    except FileNotFoundError:
        # 兼容：用户不再使用/已删除拼音词表时，保持可运行
        return set()

CHINESE_PINYIN_PATH = project_path("data", "chinese_pinyin.txt")
PINYIN_SET: set[str] = load_chinese_pinyin_from_file(CHINESE_PINYIN_PATH)

# ---------------------------------------------------------------------
# 2. 英文姓名 AC & 英文字典集合
#    • en_name_ac : 直接传给 segment_l_d_s
#    • en_dict_set: 从同一个 Aho-Corasick 自动机提取 keys() ⇒ set[str]
# ---------------------------------------------------------------------
EN_NAME_AC = load_english_name_automaton(str(project_path("data", "english_name_automaton.pkl")))

EN_DICT_AC = load_english_dict_automaton(str(project_path("data", "english_dict_automaton.pkl")))
# 预构建英文 AC：与 TrainingManager worker 初始化一致（避免每条 case 重建）
EN_AC = build_en_automaton(frozenset(EN_NAME_AC.keys()), frozenset(EN_DICT_AC.keys()))

# ---------------------------------------------------------------------
# 3. (可选) 覆盖中文姓名检测器 —— 保持与旧脚本行为一致
# ---------------------------------------------------------------------
# import training.segmenter.segment_l_d_s as lds
# class NoCNDetector:
#     def find_names(self, text):  # noqa: D401
#         return []
# lds._LDS_CN_DETECTOR = NoCNDetector()

# ---------------------------------------------------------------------
# 4. 测试用例（原封不动）
# ---------------------------------------------------------------------

test_cases = [
    ("user002@example.com", "aptx4869"),
    ("user002@example.com", "1983123"),
    ("user002@example.com", "3bsuper123"),
    ("user001@example.com", "123super"),
    ("user002@example.com", "3bsuper123"),
    ("user003@example.com", "jasonlove198715super"),
    ("user004@example.com", "victory.zhuosuper"),
    ("user005@example.com", "liyijie307super"),
    ("user006@example.com", "super123123"),
    ("user007@example.com", "yaohua.1388super"),
    ("user008@example.com", "super123"),
    ("user009@example.com", "ya357765847super"),
    ("user010@example.com", "zhangjiying80super"),
    ("user011@example.com", "superwoman"),
    ("user012@example.com", "super123"),
    ("user013@example.com", "fanxudon_2006super"),
    ("user014@example.com", "yiiy250super"),
    ("user015@example.com", "superboy0311"),
    ("user016@example.com", "xuyunhai5super"),
    ("user017@example.com", "dgs0814super"),
    ("user018@example.com", "weiyk520super"),
    ("user019@example.com", "zhencheng0111super"),
    ("user020@example.com", "lijie.0008super"),
    ("user021@example.com", "fyjk007super"),
    ("user022@example.com", "super123"),
    ("user023@example.com", "superman"),
    ("user024@example.com", "cx_warcraftsuper"),
    ("user025@example.com", "super123"),
    ("user026@example.com", "bytgmrx96super"),
    ("user027@example.com", "super1231"),
    ("user028@example.com", "super123"),
    ("user029@example.com", "gramble_2006super"),
    ("user030@example.com", "123456peterluguofusuper"),
    ("user031@example.com", "liuzihang008super"),
    ("user032@example.com", "huiyiyyz2super"),
    ("user033@example.com", "david991027super"),
    ("user034@example.com", "super123"),
    ("user035@example.com", "tboss888super"),
    ("user036@example.com", "zhouyang22004super"),
    ("user037@example.com", "shihua560412super"),
    ("user038@example.com", "super123"),
    ("user039@example.com", "ak31362164super"),
    ("user040@example.com", "superking"),
    ("user041@example.com", "frank_houxinggangsuper"),
    ("user042@example.com", "Superman.904"),
    ("user043@example.com", "fanjinliusuper"),
    ("user044@example.com", "jerryivy1314super"),
    ("user045@example.com", "super123"),
    ("user046@example.com", "yf9hsuper"),
    ("user047@example.com", "lvrui521super"),
    ("user048@example.com", "rendongcsuper"),
    ("user049@example.com", "wohenainizhuhaiyunsuper"),
    ("user050@example.com", "super123super123"),
    ("user051@example.com", "anythingformykingzisuper"),
    ("user052@example.com", "lovecuiwen2005super"),
    ("user053@example.com", "yuezflovesuper"),
    ("user054@example.com", "lz132002super"),
    ("user055@example.com", "qzplsuper"),
    ("user056@example.com", "malinzhong1super"),
    ("user057@example.com", "supergife"),
    ("user058@example.com", "super123"),
    ("user059@example.com", "super123"),
    ("user060@example.com", "ehinan1984121super"),
    ("user061@example.com", "xumulangsuper"),
    ("user062@example.com", "fay3124622super"),
    ("user063@example.com", "tlay.xinsuper"),
    ("user064@example.com", "superstar"),
    ("user065@example.com", "happy44cnsuper"),
    ("user066@example.com", "super007"),
    ("user067@example.com", "keyuanzheng888super"),
    ("user068@example.com", "wanglan0416super"),
    ("user069@example.com", "caosky83super"),
    ("user070@example.com", "heplxwsuper"),
    ("user071@example.com", "yu414721683super"),
    ("user072@example.com", "yaoyuchongsuper"),
    ("user073@example.com", "chenmeibosuper"),
    ("user074@example.com", "shaerfeisuper"),
    ("user075@example.com", "super123"),
    ("user076@example.com", "super123"),
    ("user077@example.com", "emmastellasuper"),
    ("user078@example.com", "yan520sisuper"),
    ("user079@example.com", "super1231"),
    ("user080@example.com", "katiesuper"),
    ("user081@example.com", "wuzhenjie111super"),
    ("user082@example.com", "super123"),
    ("user083@example.com", "superzp1980"),
    ("user084@example.com", "mengmeng520100super"),
    ("user085@example.com", "zy.1949super"),
    ("user086@example.com", "bhzthsuper"),
    ("user087@example.com", "wgqwangguangqiangsuper"),
    ("user088@example.com", "super123"),
    ("user089@example.com", "yanjing88811super"),
    ("user090@example.com", "xiaoboy199699super"),
    ("user091@example.com", "24super123"),
    ("user092@example.com", "zhoupipi530super"),
    ("user093@example.com", "qiujunlunsuper"),
    ("user094@example.com", "yuchao_2007super"),
    ("user095@example.com", "leansapphiresuper"),
    ("user096@example.com", "liujiangsuper"),
    ("user097@example.com", "bennieaaasuper"),
    ("user098@example.com", "super123"),
    ("user099@example.com", "fangfeimeili2008super"),
    ("user100@example.com", "wq76815super"),
    ("user001@example.com", "ZUIMANGBANG123"),
    ("user002@example.com", "KAIKAI2025"),
    ("user003@example.com", "KAIKAIFA!"),
    ("user004@example.com", "BUBUBUBU"),
    ("user005@example.com", "EIEIEI"),
    ("user006@example.com", "OUOUOU"),
    ("user007@example.com", "UAIUAIUAI"),
    ("user008@example.com", "SUANSHU88"),
    ("user009@example.com", "KUAIEN"),
    ("user010@example.com", "woaini"),
]

# ---------------------------------------------------------------------
# 5. 执行流水线
# ---------------------------------------------------------------------
for idx, (account, password) in enumerate(test_cases, 1):
    print("\n" + "=" * 60)
    print(f"[CASE {idx}]  account = {account}   password = {password}")

    # step-1 预处理 / 粗分
    coarse = preprocess_breach(account, password)
    print("[DEBUG] coarse_segments =", coarse)

    # step-2 L/D/S 并行细分
    fine = segment_l_d_s(
        coarse,
        EN_AC,
        PINYIN_SET,
        topk_l=TOPK_L,               # 与训练一致
    )
    print("[DEBUG] fine_segments   =", fine)

    # step-3 多层回溯后处理（leet + 英文名合并）
    post_paths = postprocess_multibranch(
        fine,
        EN_NAME_AC,
        EN_DICT_AC,             # 后处理仍用自动机做 membership
        expand_top_paths=EXPAND_TOP_PATHS,    # 与训练一致
        merged_top_paths=MERGED_TOP_PATHS,    # 与训练一致
    )
    # 取概率最高的一条
    # ————————————————————————————————————————————————————
    # 取前 5 条候选并计算它们进入训练器的真实权重
    # ————————————————————————————————————————————————————
    if post_paths:
        # 1) soft-max 温度缩放（与 TrainingManager 中一致）
        TEMP_ALPHA = 1.8
        lps = [lp for _, lp in post_paths]
        max_lp = max(lps)
        raw_ps = [math.exp(TEMP_ALPHA * (lp - max_lp)) for lp in lps]
        Z = sum(raw_ps) or 1.0
        posteriors = [p / Z for p in raw_ps]

        # 2) 阈值过滤 + top-K（最多保留 MAX_SELECTED_PATHS 条）
        weighted = []
        for (segs, lp), post in zip(post_paths, posteriors):
            if post < POSTERIOR_THRESHOLD:
                break
            weighted.append((segs, post))
            if len(weighted) >= MAX_SELECTED_PATHS:
                break

        # 3) 重新归一化（确保权重和为 1）
        total_w = sum(w for _, w in weighted) or 1.0
        weighted = [(segs, w / total_w) for segs, w in weighted]

        # 4) 打印：这就是最终送给 Trainer.update_counts 的 samples
        print("[DEBUG] final weighted samples for training:")
        for i, (segs, w) in enumerate(weighted, 1):
            print(f"  [{i:02d}] segments = {segs}   weight = {w:.4f}")
    else:
        print("[DEBUG] no post_paths, no samples to train.")

# test_cases = [
#     ("liyong@gmail.com", "liyong2019"),
#     ("testuser", "li@yong_2023"),
#     ("testuser", "p@$$w0rd"),
#     ("testuser", "abcabcabc"),
#     ("testuser", "qwertasdfg"),
#     ("duplicated", "1qaz2wsx"),
#     ("testuser", "peter2023peter"),
#     ("testuser", "aa"),  # 单一字符重复2次，不满足条件，不应识别为sr
#     ("testuser", "aaa"),  # 单一字符重复3次，应识别为sr3
#     ("testuser", "Ella"),  # 多字符重复2次，应识别为sr2
#     ("testuser", "ABCABC"),  # 混合大小写，重复两次，应识别为sr2（忽略大小写）
#     ("testuser", "abcdabcdabcd"),  # 多字符重复3次，应识别为sr3
#     ("testuser", "A"),
#     ("testuser", "!!!!"),  # 特殊符号，不满足纯字母条件，不进行 sr 检测
#     ("testuser", "1111"),  # 数字，不满足纯字母条件，不进行 sr 检测
#     ("testuser", "##$$##$$"),  # 特殊符号组合，不满足纯字母条件，不进行 sr 检测
#     ("testuser", "ababab"),  # 多字符重复，"ab"重复3次，应识别为 sr3
#     ("testuser", "AbAbAb"),  # 混合大小写的"ab"重复3次，应识别为 sr3
#     ("testuser", "a1a1a1"),  # 字母和数字混合，不满足纯字母条件，不进行 sr 检测
#     ("testuser", "zzzz"),  # 单一字符"z"重复4次，应识别为 sr4
#     ("testuser", "xyxy"),  # 多字符"xy"重复2次，应识别为 sr2
#     ("testuser", "Xyxyxy"),  # 多字符"xy"重复3次，应识别为 sr3
#     # 单字符出现一次，不构成重复，不应识别为sr
#     # 常见顶级域名
#     ("alice@example.com", "mypasswordexample.com"),
#     ("bob@test.org", "secureTEST.org123"),
#     ("charlie@sample.net", "charlieSAMPLE.net!"),
#     # 二级及多级域名
#     ("david@sub.domain.co.uk", "loginSub.Domain.CO.UK"),
#     ("eve@mail.service.io", "tokenMAIL.service.IOxyz"),
#     # 教育机构和国家域名
#     ("frank@university.edu.cn", "accessUNIVERSITY.edu.cn"),
#     ("grace@school.ac.jp", "GRACEschool.AC.JPpass"),
#     # 新兴及特殊顶级域名
#     ("hank@startup.tech", "inviteSTARTUP.tech2025"),
#     ("irene@my-site.online", "visitMY-SITE.ONLINE"),
#     ("jack@cool-site.club", "joinCOOL-SITE.CLUB!"),
#     ("kate@service.info", "infoSERVICE.INFOdata"),
#     ("leo@domain.ai", "useDOMAIN.AItools"),
#     ("maya@custom-domain.dev", "buildCUSTOM-domain.DEV"),
#     ("user@sub.domain.co.uk", "mypasswordsub.domain.co.uk!"),
#     ("user@sub.domain.co.uk", "mypasswordsub.domain.co!"),
#     ("user@sub.domain.co.uk", "xxxsub.domain.coukxxx"),
#     ("test", "abc1234def5678gh"),
#     # 前后缀都是字母，中间夹走位
#     ("testuser", "abc1234def"),
#     # 数字走位放前面，字母后缀
#     ("testuser", "jack"),
#     # 重复 abc 但中间插了 x
#     ("testuser", "Abcabcxabc"),
#     # 完全字母的 sr 重复检测（xyz 重复三次）
#     ("testuser", "xyzxyzxyz"),
#     # 全键盘走位（无重叠）
#     ("testuser", "zxcvbn"),
#     ("testuser", "anna"),
#     # 多段键盘走位
#     ("testuser", "1qaz2wsx3edc"),
#     # 纯字母，无走位、无重复
#     ("testuser", "abcdefgh"),
#     # 数字多次重复，但不符合 sr（含数字时不做 sr 识别）
#     ("testuser", "121212"),
#     # 混合字母+数字，小块 "a1" 重复 2 次 → sr2
#     ("dummy", "a1a1"),
#     # 混合字母+数字，小块 "a1" 重复 3 次 → sr3
#     ("dummy", "a1a1a1"),
#     # 纯数字，"123" 重复 1 次 → None
#     ("dummy", "123"),
#     # 纯数字，"111" 重复 3 次 → sr3
#     ("dummy", "111"),
#     # 纯数字，"2222" 重复 4 次 → sr4
#     ("dummy", "2222"),
#     # 纯数字，"202020" 重复 3 次 → sr3
#     ("dummy", "xhhelloowworldd"),
#     # 纯数字但重复不足 3 次 → None
#     ("dummy", "4545"),  # 虽然可按 "45" 重复两次，但纯数字需≥3
#     # 纯字母，"abab" 重复 2 次 → sr2
#     ("dummy", "abab"),
#     # 纯字母，"abcdabcd" 重复 2 次 → sr2
#     ("dummy", "helloworld"),
#     # 纯字母，"abcabcabc" 重复 3 次 → sr3
#     ("dummy", "abcabcabc"),
#     # 纯字母但单字符重复不足 3 次 → None
#     ("dummy", "zz"),  # 块 "z" 重复2次，但单字符需≥3
#     # 混合且无法整除 → None
#     ("dummy", "a1a2a1"),
#     # 混合小块 "ab1" 重复 2 次 → sr2
#     ("dummy", "ab1ab1"),
#     ("alice@mail.co.uk", "mail.co.uk123"),
#     ("alice@mail.co.uk", "mail123"),
#
#     # .edu.cn 中国教育网，多级 TLD
#     ("bob@university.edu.cn", "university.edu.cn2025"),
#     ("bob@university.edu.cn", "university2025"),
#
#     # .ac.uk 英国学术机构，多级 TLD
#     ("carol@dept.school.ac.uk", "dept.school.ac.ukpass"),
#     ("carol@dept.school.ac.uk", "deptschoolpass"),
#
#     # .co.jp 日本企业域，多级 TLD
#     ("dave@sub.example.co.jp", "sub.example.co.jp!"),
#     ("dave@sub.example.co.jp", "subexamplepwd"),
#
#     # 普通子域 + 单级 TLD
#     ("frank@sub.example.com", "example.comXYZ"),
#     ("frank@sub.example.com", "exampleXYZ"),
#     # 1. 单词 + 日期
#     ("edgecase", "hello19991231"),  # → [('hello','en_word'), ('19991231','yyyymmdd')]
#
#     # 2. 中间包含 11 位手机号
#     ("edgecase", "callme13800138000"),  # → [('call','en_word'), ('me','en_word'), ('13800138000','cn_mobile')]
#
#     # 3. 英文名首 + 拼音尾
#     ("edgecase", "Michaelzhang"),  # → [('Michael','en_name'), ('zhang','py')]
#
#     # 4. 混合大小写 + 非词典短词
#     ("edgecase", "LiGO2024"),  # → [('Li','py'), ('GO','nn') ⚠︎ len==2 被过滤, ('2024','year')]
#
#     # 5. 全大写缩写（词典无条目）
#     ("edgecase", "NASA2025!"),  # → [('NASA','nn') ⚠︎ 应考虑 acronym?, ('2025','year'), ('!','spec1')]
#
#     # 6. 典型 leet
#     ("edgecase", "Pa55w0rd!"),
#     # → [('P','nn'), ('a','nn'), ('55','number2'), ('w','nn'), ('0','nn'), ('rd','nn'), ('!','spec1')] ⚠︎ leet 未还原
#
#     # 7. 短英文词（被字典过滤）×2
#     ("edgecase", "higo"),  # → [('hi','nn'), ('go','nn')] ⚠︎ len<3 无法识别
#
#     # 8. 连续非词典字母 + 特符
#     ("edgecase", "xyz@@@123"),  # → [('xyz','nn'), ('@@@','spec3'), ('123','number3')]
#
#     # 9. 长数字但不是手机号
#     ("edgecase", "helloworld"),  # → [('123456789012','number12')]  合理
#
#     # 10. 中文拼音夹英文单词
#     ("edgecase", "woainiyadahuaidan"),
#
#     ("edgecase", "McDonald"),
#     ("edgecase", "Alicesmith"),
#
#     ("edgecase", "apple1apple"),
#     ("edgecase", "fooapplebarapple"),
#     ("edgecase", "chencheng"),
#     ("edgecase", "20250101"),     # 8 位日期，年份 > 2015（检验年份上限）
#     ("edgecase", "19990229"),     # 非闰年 2-29，非法日期
#     ("edgecase", "20000229"),     # 闰年合法 2-29
#     ("edgecase", "23101"),        # 省略 0 的 yy-m-d（23-10-1）
#     ("edgecase", "2101"),         # 长度 4，月 = 0 → 应判非日期
#     ("edgecase", "17001234567"),  # 170 号段手机号，验证号段过滤
#     ("edgecase", "13800138000"),  # 正常手机号
#     ("edgecase", "123456789012"), # 12 位纯数字，落到 number12
#     ("edgecase", "99999999"),     # 8 位非法日期，全 9
#     ("nianping@mail03.shufe.edu.cn", "@mail03.shufe.edu.cn"),     # 月 = 13，非法日期
#
# ]

# test_cases = [
#     ("130221199903020030", "sjyrdwl2!"),
#     ("2023214906@163.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906@stu.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906", "Sjyrdwl2"),
#     ("2016111761", "tp6h93zj4"),
#     ("2023665173", "Czh661896"),
#     ("abdoulrazakyameogo01@gmail.com", "Zakracity01"),
#     ("pathaweephu", "Pathaweephu1449"),
#     ("DerrerAgneta@csx.com", "9W5bkafx"),
#     ("Ugonna508", "Spikey123"),
#     ("2017310084", "wang537594"),
#     ("xavier11", "111111"),
#     ("2020212039", "Vayo2020"),
#     ("waqas8662", "Honeybee7me"),
#     ("2032954008@qq.com", "xZ5A3N9aLrS4EWy"),
#     ("nalinthip@mailwear.com", "#3Jyasxi2w7W32N"),
#     ("chutikarn@goedumail.com", "v45YGCPWSbpt8TL"),
#     ("weerawit@mailwear.com", "VV6DJyy9fihiCKi"),
#     ("kongsuwan@qenmail.com", "q38Q6ReJ"),
#     ("natnicha@mailwear.com", "HxR9TpfQbyNp3DF"),
#     ("natnicha@goedumail.com", "xZ5A3N9aLrS4EWy"),
#     ("zer4514@gmail.com", "AB4384768"),
#     ("2021110159", "lai713997"),
#     ("2023214906@sufe.edu.cn", "sjyrdwl2"),
#     ("2023214906@sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906", "sjyrdwl2"),
#     ("2023214906", "Sjyrdwl2"),
#     ("130221199903020030", "sjyrdwl2!"),
#     ("2023214906", "sjyrdwl2!"),
#     ("2023214906@163.sufe.edu.cn", "sjyrdwl2"),
#     ("2023214906@163.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906@stu.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906@stu.sufe.edu.cn", "sjyrdwl2"),
#     ("2023214906", "Sjyrdwl2"),
#     ("2023214906", "sjyrdwl2"),
#     ("2023214906", "Sjyrdwl2!"),
#     ("2023214906@163.sufe.edu.cn", "Sjyrdwl2!"),
#     ("2023214906@stu.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906", "sjyrdwl2!"),
#     ("2023214906", "Sjyrdwl2!"),
#     ("2023214906@163.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906@stu.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906@sufe.edu.cn", "sjyrdwl2"),
#     ("2023214906@sufe.edu.cn", "Sjyrdwl2"),
#     ("130221199903020030", "sjyrdwl2!"),
#     ("2023214906", "Sjyrdwl2"),
#     ("2032954008@qq.com", "xZ5A3N9aLrS4EWy"),
#     ("nalinthip@mailwear.com", "#3Jyasxi2w7W32N"),
#     ("chutikarn@goedumail.com", "v45YGCPWSbpt8TL"),
#     ("weerawit@mailwear.com", "VV6DJyy9fihiCKi"),
#     ("kongsuwan@qenmail.com", "q38Q6ReJ"),
#     ("natnicha@mailwear.com", "HxR9TpfQbyNp3DF"),
#     ("natnicha@goedumail.com", "xZ5A3N9aLrS4EWy"),
#     ("zer4514@gmail.com", "AB4384768"),
#     ("skwizeramukunzi", "7Lkb9MRRGnx3BFK"),
#     ("2017310084", "wang537594"),
#     ("Rastejah", "Jahknoxx8"),
#     ("2019111425", "01511zhouzhichao"),
#     ("LiebhardtAntonina@berryglobal.com", "kA0nN0EtedEm"),
#     ("2021212571", "oracle2015"),
#     ("Rastejah", "Jahknoxx8"),
#     ("2023214906", "Sjyrdwl2"),
#     ("2023214906@163.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906@stu.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906", "Sjyrdwl2"),
#     ("Ugonna508", "Spikey123"),
#     ("2017310084", "wang537594"),
#     ("xavier11", "111111"),
#     ("2020212039", "Vayo2020"),
#     ("waqas8662", "Honeybee7me"),
#     ("2032954008@qq.com", "xZ5A3N9aLrS4EWy"),
#     ("nalinthip@mailwear.com", "#3Jyasxi2w7W32N"),
#     ("chutikarn@goedumail.com", "v45YGCPWSbpt8TL"),
#     ("weerawit@mailwear.com", "VV6DJyy9fihiCKi"),
#     ("kongsuwan@qenmail.com", "q38Q6ReJ"),
#     ("natnicha@mailwear.com", "HxR9TpfQbyNp3DF"),
#     ("natnicha@goedumail.com", "xZ5A3N9aLrS4EWy"),
#     ("zer4514@gmail.com", "AB4384768"),
#     ("2021110159", "lai713997"),
#     ("2021110159", "Lai713997"),
#     ("2021110159", "Lai713997"),
#     ("2021110159", "lai713997"),
#     ("2021110159", "Lai713997"),
#     ("ryanryan22522", "ryanryan22522"),
#     ("2021110159", "Lai713997"),
#     ("markborboh", "Johnigirl69"),
#     ("2227222278", "ypl22222726"),
#     ("2021110159", "lai713997"),
#     ("skwizeramukunzi", "7Lkb9MRRGnx3BFK"),
#     ("2021110159", "Lai713997"),
#     ("2021110159", "lai713997"),
#     ("2021110159", "ryanryan22522"),
#     ("2021110159", "Lai713997"),
#     ("ryanryan22522", "ryanryan22522"),
#     ("2021110159", "lai713997"),
#     ("ahmed15558@hu.edu.eg", "Ahmed15558"),
#     ("fwillmore@autozone.com", "dT64bPu5izYkxcX"),
#     ("2014712134", "stea154!"),
#     ("2021212571", "oracle2015"),
#     ("2021212571", "Oracle2015"),
#     ("2023665173", "Czh661896"),
#     ("abdoulrazakyameogo01@gmail.com", "Zakracity01"),
#     ("pathaweephu", "Pathaweephu1449"),
#     ("DerrerAgneta@csx.com", "9W5bkafx"),
#     ("waqas8662", "Honeybee7me"),
#     ("xavier11", "111111"),
#     ("Ugonna508", "Spikey123"),
#     ("markborboh", "Johnigirl69"),
#     ("Rastejah", "Jahknoxx8"),
#     ("skwizeramukunzi", "7Lkb9MRRGnx3BFK"),
#     ("Samuel", "7Lkb9MRRGnx3BFK"),
#     ("13702733937", "CHENhaimei930505@"),
#     ("й™€жµ·еЄљ", "CHENhaimei930505@"),
#     ("markborboh", "Johnigirl69"),
#     ("2227222278", "ypl22222726"),
#     ("2021110159", "lai713997"),
#     ("skwizeramukunzi", "7Lkb9MRRGnx3BFK"),
#     ("2021110159", "Lai713997"),
#     ("2021110159", "lai713997"),
#     ("2021110159", "ryanryan22522"),
#     ("2021110159", "Lai713997"),
#     ("ryanryan22522", "ryanryan22522"),
#     ("2021110159", "lai713997"),
#     ("ahmed15558@hu.edu.eg", "Ahmed15558"),
#     ("fwillmore@autozone.com", "dT64bPu5izYkxcX"),
#     ("2014712134", "stea154!"),
#     ("2021212571", "oracle2015"),
#     ("2021212571", "Oracle2015"),
#     ("2023665173", "Czh661896"),
#     ("LiebhardtAntonina@berryglobal.com", "kA0nN0EtedEm"),
#     ("AldenNatassia@kindredhealthcare.com", "Tj7OxXdMW"),
#     ("Alirich", "Salo173moo"),
#     ("tanapat.lei@bumail.com", "Godly0951190266"),
#     ("2018934810", "080199"),
#     ("150928200004075443", "075443"),
#     ("FVandermolen@delta.com", "cjKHLdQQQFoiQRh"),
#     ("2023214906@163.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906@163.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906@stu.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906@stu.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906", "Sjyrdwl2"),
#     ("2023214906", "Sjyrdwl2"),
#     ("2023214906@sufe.edu.cn", "sjyrdwl2"),
#     ("2023214906@sufe.edu.cn", "sjyrdwl2"),
#     ("130221199903020030", "sjyrdwl2!"),
#     ("2023214906", "Sjyrdwl2"),
#     ("2032954008@qq.com", "xZ5A3N9aLrS4EWy"),
#     ("nalinthip@mailwear.com", "#3Jyasxi2w7W32N"),
#     ("chutikarn@goedumail.com", "v45YGCPWSbpt8TL"),
#     ("weerawit@mailwear.com", "VV6DJyy9fihiCKi"),
#     ("kongsuwan@qenmail.com", "q38Q6ReJ"),
#     ("natnicha@mailwear.com", "HxR9TpfQbyNp3DF"),
#     ("natnicha@goedumail.com", "xZ5A3N9aLrS4EWy"),
#     ("zer4514@gmail.com", "AB4384768"),
#     ("faiyadfarebisiam23@gmail.com", "moraadul07"),
#     ("faiyadfarebisiam23@gmail.com", "moraadul07"),
#     ("angelkachipapa@gmail.com", "angelline1988"),
#     ("alexhales86@outlook.com", "hazrats11"),
#     ("andrewhalewood@hotmail.com", "mythreythomas007"),
#     ("andrewhalewood@hotmail.com", "mythreythomas007"),
#     ("kavumatimothy1@gmail.com", "Godschosen1"),
#     ("kavumatimothy1@gmail.com", "Godschosen1"),
#     ("sylvestersimendi90@yahoo.com", "Simendi1990"),
#     ("sylvestersimendi90@yahoo.com", "Simendi1990"),
#     ("tickimit@hotmail.com", "sirgee45mad"),
#     ("tickimit@hotmail.com", "sirgee45mad"),
#     ("obiriisaac", "jinijini1"),
#     ("juancross@gmail.com", "junior8699"),
#     ("juancross@gmail.com", "junior8699"),
#     ("ellaford99@yahoo.com", "chibuzor"),
#     ("ellaford99@yahoo.com", "chibuzor"),
#     ("kavumatimothy1@gmail.com", "Godschosen1"),
#     ("abozlatan512@yahoo.com", "zlatanmega773##"),
#     ("yehezkitishrey@gmail.com", "Shrey12345"),
#     ("yehezkitishrey@gmail.com", "Shrey12345"),
#     ("mugumeernest2015@gmail.com", "EDMOND45"),
#     ("mugumeernest2015@gmail.com", "EDMOND45"),
#     ("molonalberto95@gmail.com", "bezawit21$"),
#     ("molonalberto95@gmail.com", "bezawit21$"),
#     ("rubenpe@ceipsanchisguarnerelx.com", "bezawit21$"),
# 	("sophie5951", "Pes3485aresi"),
#     ("rubenpe@ceipsanchisguarnerelx.com", "bezawit21$"),
#     ("2013210570", "183535"),
#     ("ugonna508", "spikey123"),
#     ("370502199902212833", "Yx1bwcx8!"),
#     ("34080219970403001x", "jwd5568008"),
#     ("jwd", "jwd5568008"),
#     ("lljszzc@163.sufe.edu.cn", "chaochao666"),
#     ("Carma.Chey@publix.com", "rJVxrI3qU793eyL"),
#     ("FeaganesMerline@united.com", "Eanj22cO55pz4"),
#     ("2021110159", "lai713997"),
#     ("zhucungen", "6860654gen"),
#     ("lljszzc.edu.cn", "chaochao666"),
#     ("FlemmonsMarcelle", "oPxtwjOmYdOZ"),
#     ("2009115091", "ft901118"),
#     ("2018111107", "57838132egoist"),
#     ("130221199903020030", "sjyrdwl2!"),
#     ("haseen azam", "has12116102"),
#     ("Alirich", "Salo173moo"),
#     ("3121222738", "12822zhouzhichao"),
#     ("2019110474", "xxbghhc0"),
#     ("1010601002", "Ices012345"),
#     ("Denni.Campain@aes.com", "TslCCR0Z0gb2"),
#     ("Shen@163.sufe.edu.cn", "Syy20030711"),
#     ("amine.bairouk", "Q3f2P6sEtFf7qRP"),
#     ("Abdirashid", "g4549757"),
#     ("mariashabbir437@gmail.com", "mariaa123"),
#     ("Felice.Ramsy@esteelauder.com", "wHQ1e9qg"),
#     ("2020211157", "qwerty11"),
#     ("2020712003", "Ices123456"),
#     ("ntiakomensah", "EVELYNB34"),
#     ("65230119930715081X", "houati1993715"),
#     ("MBewley@abbvie.com", "OQETdTAI"),
#     ("Samuel", "7Lkb9MRRGnx3BFK"),
#     ("bkalahiki@interpublic.com", "j5pcc42nQkEHdy"),
#     ("Fisha Semaw", "Fish@#12"),
#     ("LehnertzAlysia@campbells.com", "YyL9Qb6TK"),
#     ("Youshaa Alhorani", "Y98oushaa34"),
#     ("Mark Borboh", "Johnigirl69"),
#     ("George Aggrey", "0542037470aA"),
#     ("2012720608", "Wu19941120"),
#     ("603033199502082216", "bano8592834"),
#     ("2018111107", "egoist57838132"),
#     ("haseen azam", "has12116102"),
#     ("ugonna508", "spikeY123"),
#     ("2021211172", "dangwei0519"),
#     ("2020210615", "Hsj3619469"),
#     ("2018110203", "2018110203"),
#     ("2020110715", "2020110715"),
#     ("2020210378", "zqm20010927"),
#     ("573321199905091323", "TTyy05091323"),
#     ("370724200001210060", "songchunlei12"),
#     ("2016110214", "qy920y68s72@$"),
#     ("2019219009", "sufe0112819"),
#     ("Fisha Semaw", "Fish@#12"),
#     ("2018111107", "57838132egoist"),
#     ("haseen azam", "has12116102"),
#     ("Mark Borboh", "Johnigirl69"),
#     ("George Aggrey", "0542037470aA"),
#     ("Bezawit gezahegne", "bezawit21$"),
#     ("2023214906", "Sjyrdwl2"),
#     ("2022722529", "5449393102Aa"),
#     ("2022665339", "Ljw86520008"),
#     ("2021110159", "lai713997"),
#     ("ugonna508", "spikey123"),
#     ("zhucungen", "6860654gen"),
#     ("lljszzc@163.sufe.edu.cn", "chaochao666"),
#     ("2014213751", "681217"),
#     ("241002199808220012", "zrx19980822"),
#     ("2005111235", "mima2005111235"),
#     ("BGallian@scotiabank.com", "47ryziSR"),
#     ("nkidner@visa.com", "1fyRpsbO"),
#     ("wenyimeng@morganstanley.com", "KW6Yw1j7"),
#     ("nmbolu@cisco.com", "5kOi4NfK"),
#     ("Youshaa Alhorani", "Y98oushaa34"),
#     ("youshaa.alhorani@hotmail.com", "Y98oushaa34"),
#     ("MMaltezo@bigpoint.com", "Ddp0diNN"),
#     ("417902199506232713", "qingzhi19950623"),
#     ("45030419971015153X", "ohS8xC"),
#     ("045541Zhang", "zhang045541"),
#     ("2019110474", "xxbghhc0"),
#     ("2010110359", "GUO19921111"),
#     ("2019110055", "lr941128"),
#     ("39090219850514186X", "19850514"),
#     ("19930405", "abcd1993"),
#     ("211283199404212522", "19940421"),
#     ("440981199411021648", "19941102"),
#     ("620523199804021658", "19980402"),
#     ("38070119940425001X", "19940425"),
#     ("342201199501181230", "19950118"),
#     ("450602199901200068", "19990120"),
#     ("41152519960604001X", "19960604"),
#     ("362526198910040047", "19891004"),
#     ("360429199101170051", "19910117"),
#     ("411527199811043958", "19981104"),
#     ("330382199209250090", "19920925"),
#     ("362201199109294526", "19910929"),
#     ("320211199605102035", "19960510"),
#     ("130928199302150071", "19930215"),
#     ("130703199404080222", "19940408"),
#     ("652826197506080047", "19750608"),
#     ("152801199308110020", "19930811"),
#     ("320324199502112138", "19950211"),
#     ("210381198711056020", "19871105"),
#     ("152801199706284236", "19970628"),
#     ("654003199210192731", "19921019"),
#     ("652801198707044629", "19870704"),
#     ("652801198707044629", "19870704"),
#     ("652801198707044629", "abcd19870704"),
#     ("654003199210192731", "19921019"),
#     ("654003199210192731", "Abcd19921019"),
#     ("652801199308110020", "19930811"),
#     ("652801199308110020", "Abcd19930811"),
#     ("654003199210192731", "19921019"),
#     ("654003199210192731", "abcd19921019"),
#     ("652801199308110020", "Abcd19930811"),
#     ("652801199308110020", "19930811"),
#     ("Ztitus", "Titus119"),
#     ("czer0", "czer0516"),
#     ("puifai95", "puifai95123"),
#     ("Zhucungen", "6860654gen"),
#     ("lljszzc.edu.cn", "chaochao666"),
#     ("652801199308110020", "Abcd19930811"),
#     ("Zdenka", "Zdenka123"),
#     ("LernerKyung", "Kyung!0901"),
#     ("MeddersZack", "Med@1976"),
#     ("Nandkumar", "nandu@2012"),
#     ("nkidner", "nkidner@"),
#     ("RHamilton", "rlH8739!"),
#     ("MPortelli", "Mp001122@@"),
#     ("vbowin@21cf.com", "9MbaTkEkl5p2"),
#     ("zrials@morganstanley.com", "ENgC0VtCcN"),
#     ("aggrey", "0542037470aA"),
#     ("ghaffar", "Statistician"),
#     ("vbowin", "9MbaTkEkl5p2"),
#     ("zrials", "ENgC0VtCcN"),
#     ("2009115091", "ft901118"),
#     ("2019219009", "sufe0112819"),
#     ("2011110156", "Lai713997"),
#     ("2016110214", "qy920y68s72@$"),
#     ("XRabenecker", "aaaaaa6"),
#     ("avaughn", "21158720"),
#     ("BHammon", "390556"),
#     ("TMynatt", "464646"),
#     ("ORahaim", "rahim123"),
#     ("YBeffa", "beffa321"),
#     ("KLeatherbury", "111111"),
#     ("345555198912341234", "abc12345"),
#     ("Y98oushaa34", "Y98oushaa34"),
#     ("Fish@#12", "Fish@#12"),
#     ("6860654gen", "6860654gen"),
#     ("ljw86520008", "Ljw86520008"),
#     ("sjyrdwl2", "Sjyrdwl2"),
#     ("2012720608", "Wu19941120"),
#     ("2012720608", "Wu19941120"),
#     ("2012720608", "Wu19941120"),
#     ("Wchenshe", "Wh3nSh3Sm1l3S"),
#     ("Ntian", "Ntiamwen1967"),
#     ("Bmueller", "Xxprince21!"),
#     ("stTrujillo", "TrujilloStu0929"),
#     ("rProcknow", "Pxq098"),
#     ("Bsaah", "654321Ab"),
#     ("Nessie", "Lockness"),
#     ("19841122", "Abcd1122"),
#     ("EFeight", "EF8eight"),
#     ("CQuilico", "cquilico"),
#     ("TNewhouse", "newhouse"),
#     ("IObrien", "iobrien"),
#     ("DLenk", "dlenk321"),
#     ("Schwoops", "Schwoops123"),
#     ("CWehr", "wehr345"),
#     ("SBeloat", "beloat!"),
#     ("TPentland", "pentland123"),
#     ("GHoupt", "ghoupt987"),
#     ("EAbernathy", "eabernathy"),
#     ("RJinright", "jinright12"),
#     ("JMaximin", "jmaximin"),
#     ("Esudler", "987654321"),
#     ("MCoderre", "Mcoderre!23"),
#     ("FMutone", "fmutone99"),
#     ("JKillingsworth", "JKill123"),
#     ("EPeffley", "peffley@"),
#     ("ASlangen", "aslangen##"),
#     ("MGravalec", "mgravalec"),
#     ("JBennette", "jbennette1"),
#     ("rcallen", "calle!42"),
#     ("239231199106016067", "19910601"),
#     ("652801199308110020", "Abcd19930811"),
#     ("381102199409123218", "940912"),
#     ("2009115091", "ft901118"),
#     ("652801199308110020", "19930811"),
#     ("382101198712152718", "19980508"),
#     ("652801199308110020", "Abcd19930811"),
#     ("2009115091", "ft901118"),
#     ("652801199308110020", "19930811"),
#     ("382101198712152718", "19980508"),
#     ("652801199308110020", "Abcd19930811"),
#     ("652801199308110020", "19930811"),
#     ("Vbennette", "bennette23"),
#     ("ZLeja", "myzd75!"),
#     ("LKirchner", "LKirc890"),
#     ("Ctony", "tony03"),
#     ("Nnorris", "norris99"),
#     ("aakerman", "aakerman12"),
#     ("jbarnum", "barnum1970"),
#     ("jsherrell", "sherrell5"),
#     ("dzima", "dzima@0"),
#     ("bjorn", "bjorn#321"),
#     ("sguerriero", "sguerriero"),
#     ("dvesser", "dvesser!"),
#     ("cyohn", "Cyohn123"),
#     ("dlonsway", "dlonsway"),
#     ("mpfrappier", "mpfrappier$"),
#     ("kfukui", "kfukui##"),
#     ("mosunwo", "mosunwo"),
#     ("msechrest", "msechrest!"),
#     ("ragosnell", "ragosnell%1"),
#     ("cgennarelli", "cgennarelli@"),
#     ("bkinnaird", "bkinnaird"),
#     ("cbozeman", "cbozeman##"),
#     ("KGunn", "Kgun123"),
#     ("Zsofi", "Zsofi1"),
#     ("LBAKER", "lCBaker50"),
#     ("BRanallo", "branallo21"),
#     ("CHartsoch", "chartsoch"),
#     ("CDiiorio", "cdiiorio!"),
#     ("kchelette", "kchelette"),
#     ("kchi", "kchi90"),
#     ("kkamler", "kkamler"),
#     ("kdeak", "kdeak"),
#     ("kmilord", "kmilord"),
#     ("19731123", "abcd1123"),
#     ("abida", "abida21"),
#     ("jboyer", "JBoyer001"),
#     ("KMcginnis", "kmcginnis!"),
#     ("HAbbey", "HAbbey007"),
#     ("2009115091", "ft901118"),
# 	 ("2023214906@stu.sufe.edu.cn", "Sjyrdwl2"),
#     ("rowanzhao", "123456"),
#     ("2021110159", "Lai713997"),
#     ("rehankhalidqadri@gmail.com", "Rehan#03"),
#     ("2023214906", "Sjyrdwl2"),
#     ("LulowAnn@danaher.com", "wgKP42LjsvSoG"),
#     ("2018111775", "Narno1071162"),
#     ("2012310046", "lx224946"),
#     ("2018110140", "zhutianyi123"),
#     ("bezawitgezahegne@gmail.com", "bezawit21$"),
#     ("BKalahiki@interpublic.com", "j5pcc42nQkEHdy"),
#     ("FWillmore@autozone.com", "dT64bPu5izYkxcX"),
#     ("2019712069", "Sansansan16"),
#     ("tanapat.lei@bumail.com", "Godly0951190266"),
#     ("joeltabarara@yahoo.com", "107408"),
#     ("tanapat.lei@bumail.net", "Godly0951190266"),
#     ("2018722570@live.sufe.edu.cn", "Minhanh@2412"),
#     ("2019110474", "yxhhhk555LXL"),
#     ("2019110474", "xxbghhc0"),
#     ("alamgir", "interservices"),
#     ("markborboh", "Johnigirl81"),
#     ("310999655", "Marcel1990"),
#     ("310999655", "qwerty11"),
#     ("student", "123456"),
#     ("2014211417", "Asdfgh123"),
#     ("2014211417", "050511"),
#     ("ugonna508", "Spikey123"),
#     ("Steven Shimirwa", "Sms#123456"),
#     ("mokhele_mj904", "#12313212mj"),
#     ("310108199701160534", "cxy119868"),
#     ("320721199204104617", "wuliangyu"),
#     ("2015110114", "cxy119868"),
#     ("2016110214", "qy920y68s72@$"),
#     ("boatengsolomon840@gmail.com", "0543341960So"),
#     ("mariashabbir437@gmail.com", "mariaa123"),
#     ("bestboynoob200", "zerzura00"),
#     ("bezawit21$", "bezawit21$"),
#     ("Тест", "Тест"),
#     ("cxy119868", "cxy119868"),
#     ("pkh", "pkh"),
#     ("вЂ™Р’В·", "cxy889888"),
# 	("abdoulrazakyameogo01@gmail.com", "Zakracity01"),
#     ("2023665173", "Czh661896"),
#     ("2022665339", "Ljw86520008"),
#     ("130221199903020030", "sjyrdwl2!"),
#     ("2023214906", "sjyrdwl2"),
#     ("2023214906", "Sjyrdwl2"),
#     ("2023214906@stu.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906@163.sufe.edu.cn", "Sjyrdwl2"),
#     ("2023214906@sufe.edu.cn", "sjyrdwl2"),
#     ("Alirich", "Salo173moo"),
#     ("Ugonna508", "Spikey123"),
#     ("2022722529", "5449393102Aa"),
#     ("2015110114", "cxy119868"),
#     ("2018111775", "Narno1071162"),
#     ("KampmannMoira@libertyinteractive.com", "1HDEpXDZJlPR"),
#     ("LehnertzAlysia@campbells.com", "YyL9Qb6TK"),
#     ("GeorgeAggrey", "0542037470aA"),
#     ("KernsMaddalena@walgreensbootsalliance.com", "d7rZIXzH09KbqRU"),
#     ("Averil.Kaltenbach@cinfin.com", "kCNpEqxRwmKp"),
#     ("Emlynne.Lederhos@pseg.com", "yzv2aCqeGYhKqC"),
#     ("Ange.Alvira@centene.com", "V3qWDLbkRk8gP3O"),
#     ("angelkachipapa@gmail.com", "angelline1988"),
#     ("Abdirashid", "g4549757"),
#     ("Anlamgirgir", "interservices"),
#     ("arakib42", "Rkb202185+"),
#     ("hasdan2k5", "Seunara1"),
#     ("skwizeramukunzi", "7Lkb9MRRGnx3BFK"),
#     ("Rastejah", "Jahknoxx8"),
#     ("ntiakomensah", "EVELYNB34"),
#     ("davidk", "kingsley1"),
#     ("markborboh", "Johnigirl69"),
#     ("semawfisha", "339299"),
#     ("06432469", "amcshn63"),
#     ("2019110474", "yxhhhk555LXL"),
#     ("2021110159", "lai713997"),
#     ("13702733327", "ynq8023@$"),
#
#     ### Real cases
#
#
#     ("2023111764", "shr041230@"),
#     ("2023111851", "050813qyz!"),
#     ("2022110936", "1159363575a!"),
#     ("2024110864", "Zm2667441181@qq"),
#     ("2022110935", "lyzzdyn147258369"),
#     ("2023120020", "tj16882004!"),
#     ("2022111746", "MoMo0316~~"),
#     ("2023110768", "Gzh686868!"),
#     ("2022111578", "sjj@002906"),
#     ("2022110573", "LI20040720@"),
#     ("2024111623", "@4X7GTAhmmf"),
#     ("2022110869", "Xiaohu666!"),
#     ("2022712007", "Abc123456789!"),
#     ("2022110094", "ljynb@369955487"),
#     ("2024111097", "Zhaozhian@123456"),
#     ("2023111352", "666666QQQQQ$"),
#     ("2024111245", "Yyx117307@"),
#     ("2023110710", "gyh18789123681@"),
#     ("2024110656", "cogpadkebt$$1004"),
#     ("2024110656", "abc123"),
#     # ① 左右均非字母 → 应标 alpha4
#     ("2025110001", "abcd1234"),
#
#     # ② ”abc“ 左字母、右字母 → 不标 alpha
#     ("2025110002", "Truabcxyz"),
#
#     # ③ “xyz” 右边是 '-' → alpha3；后跟数字
#     ("2025110003", "xyz-456"),
#
#     # ④ “xyz” 左边是数字 → alpha3
#     ("2025110004", "456xyz"),
#
#     # ⑤ “mno” 左越界、右是 '!' → alpha3
#     ("2025110005", "mno!890"),
#
#     # ⑥ “stu” 左字母、右数字 → 不标 alpha
#     ("2025110006", "student123"),
#
#     # ⑦ “abcde” 左数字、右特殊符 → alpha5
#     ("2025110007", "000abcde$"),
#
#     # ⑧ 连续重复数字（sr）+ alpha3
#     ("2025110008", "888888klm@"),
#
#     # ⑨ 键盘路径 kb（无 alpha 命中）
#     ("2025110009", "qwertasdf"),
#
#     # ⑩ 纯数字段（无 kb 检测）+ alpha4
#     ("2025110010", "pqrs987654"),
# ]
