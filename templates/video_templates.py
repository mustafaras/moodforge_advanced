# templates/video_templates.py
import random

EMOTIONS = ["mutluluk", "üzüntü", "öfke", "kaygı", "nötr"]

def get_video_emotion_scores(dominant_emotion):
    return {
        "dominant_emotion": dominant_emotion,
        "emotion_scores": {
            e: round(random.uniform(0.5, 1.0), 2) if e == dominant_emotion else round(random.uniform(0.0, 0.4), 2)
            for e in EMOTIONS
        }
    }


VIDEO_TEMPLATES = {
    "Depresyon": [
        "Yüzde belirgin bir donukluk ve göz temasından kaçınma gözleniyor. Dudaklar aşağı doğru bükülmüş. Yavaş göz kırpmalar.",
        "Gözler yere sabitlenmiş, ifade minimal. Çene düşük, alın kırışık. Tüm yüz yorgun ve çökmüş bir görünümde.",
        "Kaşlar ortada birleşmiş, göz altlarında koyuluk belirgin. Mimiklerde düşük enerji izlenimi var.",
        "Ağız kapalı ve hafif açık arasında gidip geliyor. Hareket az, baş eğik. Göz çevresinde kas tonusu zayıf.",
        "Yüzde sık tekrarlanan derin nefes alışları ve göz kaçırma davranışı. Donuk bakışlar mevcut.",
        "Konuşurken dudaklar titrek, gözler dolmuş gibi. Yüzdeki kaslar gevşek. İfade durağan.",
        "Duygusal tepki neredeyse sıfır. Baş hafif öne eğilmiş. Duruş sarkık. Yanıt gecikmeli.",
        "Zaman zaman gözler kapanıyor gibi, yorgunluk belirtisi. Kaşlar sürekli çatık. Ağızda asimetri.",
        "Yanak kaslarında aktivite çok az. Göz çevresi düşük. Genelde yüz durağan pozisyonda kalıyor.",
        "Kaçınan bakışlar, gözyaşına direnme jesti. Zayıf baş sallamaları. Göz teması minimum düzeyde."
    ],
    "Bipolar": [
        "Yüzde geniş bir gülümseme, kaşlar havada. Gözler ışıldıyor. Jestlerde aşırı hareketlilik.",
        "Mimikler hızlı ve abartılı. Göz teması yoğun. Sık sık ani baş hareketleri.",
        "Gülümseme yüzün tamamını kaplıyor. Ellerde tempo tutma veya oynama davranışı görülüyor.",
        "Konuşurken mimikler konuşmanın önünde gidiyor. Jestlerle ifadeyi destekleme çok yoğun.",
        "Göz çevresinde sık sık kaş kaldırma. İfade yüksek coşku içeriyor.",
        "Yüz ifadesi ani değişiyor. Gülme ile ciddiyet arasında gidip geliyor.",
        "Baş sürekli hareket halinde. Gözlerde hiperaktiflik. Gülme krizine benzer mimikler.",
        "Yüz ifadesi sahne anlatan bir oyuncu gibi dramatik. Kaşlar yukarıda, ağız açık.",
        "Göz teması kararlı ve yoğun. Yüzde gergin olmayan ama hareketli bir enerji var.",
        "Hızlı konuşmaya eşlik eden yüz animasyonu. Gözler büyük, alın açık, dudaklar aktif."
    ],
    "Psikotik": [
        "Yüzde tedirginlik. Gözler hızlıca sağa sola hareket ediyor. Sık sık omuz üzerinden bakma.",
        "Mimikler birbirini takip etmiyor. Anlamsız sırıtma ve göz kaçırmalar arka arkaya geliyor.",
        "Gözler bir noktaya sabitlenmiş, ama kişi o noktada olmayan biriyle iletişimde gibi.",
        "Kaşlar sürekli hareket ediyor, ancak ifade anlamsız. Zaman zaman boşluğa gülme.",
        "İfade değişimleri ani ve uyumsuz. Yüz kaslarında düzensiz kasılmalar.",
        "Kendi kendine mırıldanma sırasında yüz ifadesi çevreyle bağlantısız. Gözlerde garip parıltı.",
        "Alnın ortasında kırışıklık, çenede kasılma. Göz temasından kaçınma ama aniden dik dik bakma.",
        "Baş hafif eğik, mimikler yüzün bir yarısında yoğun. Gözlerde korku ve kuşku.",
        "Yüzde anlamlandırılamayan ifadeler. Zaman zaman ağız hareketleri sanki biriyle konuşuyormuş gibi.",
        "Yüz kasları sürekli uyarılmış gibi. Rahatsız edici bir gülümseme ve ani donmalar var."
    ],
    "Anksiyete": [
        "Yüzde endişeli bir ifade. Kaşlar birbirine yaklaşmış. Dudaklar sıkıca kapalı.",
        "Göz çevresi gergin, sık göz kırpma. Yüzde kasılmalar. Baş aşağıya eğik.",
        "Elleri yüzüne götürme davranışı sık. Göz teması kesik kesik. Kaşlar hafif yukarı kalkmış.",
        "Sürekli çevreyi tarayan bakışlar. Dudak kenarları aşağıya doğru. Gözlerde telaş.",
        "Ağız kuruluğu belirtileri. Dudak yalama. Kaş çatık. Çene titrek.",
        "Hızlı mimik değişimleri. Gözlerde kararsızlık. Alnın ortası kırışık.",
        "Başta mikro sarsıntılar. Nefes alış veriş hızlanmış. Yüz kaslarında kasılma.",
        "Sık sık göz devinimi. Göz teması kurmakta zorluk. Kaşlarda hafif titreme.",
        "Yüz kaslarında aşırı tetikte olma hali. Zaman zaman yüzü silme hareketi.",
        "Yanaklarda kızarıklık. Gözler büyümüş. Kaşlar sabit düşük pozisyonda."
    ],
    "TSSB": [
        "Yüzde şok sonrası donukluk var. Gözler uzaklara dalmış. Mimik yok denecek kadar az.",
        "Zaman zaman irkilme hareketi. Gözleri bir noktada sabitleyip oraya kilitleniyor.",
        "Başka bir zaman dilimindeymiş gibi boşlukta gezinme bakışları. Kaslar gergin.",
        "Yüzde ani bir gerilim ve ardından gelen kas yumuşaması. Gözlerde yaşlılık belirtisi.",
        "Göz kırpmalar yavaş ve düşünceli. Alnın ortasında kırışıklık belirgin.",
        "Göz teması kurulsa da uzun sürmüyor. Yüzde geçmişe dönmüş bir ifade hakim.",
        "Tetikte bir yüz görünümü. Gözler çevreyi tarıyor. Kaşlar sürekli yukarıda.",
        "Sık nefes alıp verme sırasında yüz kasları istemsizce geriliyor.",
        "Gözlerde korku dolu bakış. Konuşmasız ama yoğun mimik değişimleri var.",
        "Sanki konuşsa ağlayacak gibi bir yüz ifadesi. Gözler ıslak, dudak titriyor."
    ],
    "OKB": [
        "Yüzde tedirginlik ve kararsızlık. Gözler hızlıca etrafı kontrol ediyor.",
        "Kaşlar yukarıda ve ağız kenarları gergin. Göz kırpmalar normalden fazla.",
        "Elleri yüze götürme, alnı silme hareketleri. Gözlerde yoğun dikkat hali.",
        "Yüzde sürekli bir 'emin olamama' ifadesi. Gözlerde kısa süreli panik parlamaları.",
        "Mimikler simetrik olmaya çalışıyor gibi. Kaşlar simetrik, ama gerginlik yoğun.",
        "Dudaklar içe çekilmiş. Gözler dar bir noktaya sabit. Alın kasılmış.",
        "Kaş çatma + göz kaçırma davranışı birlikte görülüyor. Yüzde sabit endişe hali.",
        "Kafasını iki yana oynatma. Yüzde hızlı mimik tekrarları.",
        "Göz temasına giriyor ama sonra hemen kaçırıyor. Ağız hareketleri tekrarlayıcı.",
        "Yüzde kontrollü bir düzeltme çabası. Elleriyle saç düzeltme, kaşlarını kontrol etme."
    ],
    "Normal": [
        "Yüz ifadesi dengeli, kaşlar ve dudaklar nötr pozisyonda.",
        "Göz teması var. Ne abartılı ne de donuk. Mimiklerde orta seviye aktivite.",
        "Yüz kasları simetrik ve rahat. Duygusal olarak dengeli bir görünüm var.",
        "Gülümseme hafif ve doğal. Göz çevresi gergin değil. Genel rahatlık hakim.",
        "Baş ve yüz hareketleri uyumlu. Stres veya taşkınlık gözlenmiyor."
    ]

}

def get_video(disorder_type):
    return VIDEO_TEMPLATES.get(disorder_type, VIDEO_TEMPLATES["Normal"])
