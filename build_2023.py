#!/usr/bin/env python3
import sqlite3, json, re, os, datetime
from collections import defaultdict, Counter
import xlsxwriter

DB='/Users/vickhung/Desktop/ig_2023_scrape/scrape.db'
OCR='/Users/vickhung/Desktop/ig_2023_scrape/ocr_cache.json'
OUT='/Users/vickhung/Documents/IG_Evidence'
os.makedirs(OUT, exist_ok=True)

DATE_START = datetime.datetime(2023,4,1, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
DATE_END   = datetime.datetime(2024,1,1, tzinfo=datetime.timezone(datetime.timedelta(hours=8)))
HKT = datetime.timezone(datetime.timedelta(hours=8))

# Method patterns
METHODS = [
  ('PayMe', r'payme'),
  ('FPS / 轉數快', r'fps|轉數快'),
  ('HSBC / 匯豐', r'hsbc|匯豐'),
  ('BEA / 東亞', r'bea\b|東亞|bank of east asia'),
  ('BOC / 中銀', r'boc\b|中銀'),
  ('Hang Seng / 恒生', r'hang\s*seng|恒生'),
  ('Standard Chartered / 渣打', r'standard chartered|渣打'),
  ('DBS', r'\bdbs\b'),
  ('Citibank / 花旗', r'citibank|花旗'),
  ('Wise', r'\bwise\b'),
  ('PayPal', r'paypal'),
  ('Alipay / 支付寶', r'alipay|支付寶'),
  ('WeChat Pay / 微信', r'wechat\s*pay|微信'),
  ('QR Code Pay', r'qr\s*code'),
  ('Bank Transfer / 轉賬', r'bank\s*transfer|轉賬|過數|入數'),
  ('Red packet 紅包/利是', r'red\s*packet|紅包|利是'),
]
GENERIC_RECEIPT = re.compile(r'已成功|款項|transfer\s*successful', re.I)
AMOUNT_RE = re.compile(r'(?:HK\$?|HKD|\$)\s?[\d,]+(?:\.\d+)?|\b[\d,]{3,}(?:\.\d{2})\b', re.I)
ACCT_RE   = re.compile(r'\b\d{8,15}\b')
PHONE_RE  = re.compile(r'\b[569]\d{7}\b')
CARD_RE   = re.compile(r'\b\d{4}\b')
TXN_RE    = re.compile(r'\d{10,}')

def detect(text, allow_generic=False):
    if not text: return [], [], [], [], [], []
    t = text
    methods = []
    for name, pat in METHODS:
        if re.search(pat, t, re.I):
            methods.append(name)
    if allow_generic and GENERIC_RECEIPT.search(t):
        methods.append('Generic Receipt')
    amounts = list(dict.fromkeys(AMOUNT_RE.findall(t)))
    accts = list(dict.fromkeys(ACCT_RE.findall(t)))
    phones = list(dict.fromkeys(PHONE_RE.findall(t)))
    cards = list(dict.fromkeys(CARD_RE.findall(t)))
    txns = list(dict.fromkeys(TXN_RE.findall(t)))
    # remove phones from accts (overlap)
    accts = [a for a in accts if a not in phones]
    return methods, amounts, accts, phones, cards, txns

def fmt_time(ts_us):
    dt = datetime.datetime.fromtimestamp(ts_us/1_000_000, tz=datetime.timezone.utc).astimezone(HKT)
    return dt.strftime('%Y-%m-%d %H:%M:%S'), dt

print('Loading DB...')
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

threads = {r['thread_id']: r['title'] or '' for r in conn.execute('SELECT thread_id, title FROM thread_queue')}
print(f'Threads: {len(threads)}')

# downloads: item_id -> file_path
dl_by_item = {}
for r in conn.execute('SELECT item_id, file_path FROM downloads WHERE status="ok"'):
    dl_by_item[r['item_id']] = r['file_path']

# OCR cache: file_path -> text
ocr = json.load(open(OCR))
print(f'OCR entries: {len(ocr)}')

# Pre-load all messages in date range
msgs = []
for r in conn.execute('SELECT thread_id,item_id,timestamp_us,user_id,username,item_type,text,media_url FROM messages'):
    ts = r['timestamp_us']
    if ts is None: continue
    _, dt = fmt_time(ts)
    if not (DATE_START <= dt < DATE_END): continue
    msgs.append(dict(r))
print(f'Messages in 2023 window: {len(msgs)}')

# ===== Build Photo Payment Evidence =====
photo_evidence = []  # rows
all_photos_in_payment_threads_candidates = []  # all photo msgs (with optional OCR data)
threads_with_payment = set()
total_photos_ocrd = 0
ocr_text_by_msg = {}  # item_id -> ocr text

# All photo-type messages
photo_msgs = [m for m in msgs if m['item_type'] in ('media','raven_media','animated_media')]
print(f'Photo-like msgs: {len(photo_msgs)}')

method_counts_evidence = Counter()
strong_count = 0
senders_payment_photos = Counter()
senders_strong = Counter()
senders_amounts = defaultdict(list)
senders_id = {}

for m in photo_msgs:
    fp = dl_by_item.get(m['item_id'])
    text = ocr.get(fp) if fp else None
    if text:
        total_photos_ocrd += 1
        ocr_text_by_msg[m['item_id']] = text
    methods, amounts, accts, phones, cards, txns = detect(text, allow_generic=True) if text else ([],[],[],[],[],[])
    has_evidence = bool(methods or amounts or accts or txns)
    if has_evidence:
        strong = bool(methods) and bool(amounts or accts or txns)
        time_str, _ = fmt_time(m['timestamp_us'])
        sender = m['username'] or str(m['user_id'])
        sid = str(m['user_id'])
        title = threads.get(m['thread_id'], '')
        photo_evidence.append([
            time_str, sender, sid, title, m['thread_id'],
            ', '.join(methods) or None,
            ', '.join(amounts) or None,
            ', '.join(accts) or None,
            ', '.join(cards) or None,
            ', '.join(phones) or None,
            ', '.join(txns) or None,
            'YES' if strong else 'NO',
            text,
            m['media_url'],
        ])
        threads_with_payment.add(m['thread_id'])
        for mm in methods: method_counts_evidence[mm]+=1
        if strong: strong_count += 1
        senders_payment_photos[sender]+=1
        if strong: senders_strong[sender]+=1
        senders_id[sender]=sid
        if amounts: senders_amounts[sender].extend(amounts[:3])

# Also: text msgs mentioning payment also mark threads_with_payment
text_msgs = [m for m in msgs if m['item_type']=='text']
print(f'Text msgs: {len(text_msgs)}')
mentions_rows = []
mention_method_counts = Counter()
mention_strong = 0
mention_threads = set()

for m in text_msgs:
    txt = m['text'] or ''
    if not txt.strip(): continue
    methods, amounts, accts, phones, cards, txns = detect(txt, allow_generic=False)
    # criterion: any method, OR amounts+accts (loose). Reference uses presence of payment keywords.
    if not methods:
        continue
    strong = bool(methods) and bool(amounts or accts or txns)
    time_str, _ = fmt_time(m['timestamp_us'])
    sender = m['username'] or str(m['user_id'])
    sid = str(m['user_id'])
    title = threads.get(m['thread_id'], '')
    mentions_rows.append([
        time_str, sender, sid, title, m['thread_id'],
        ', '.join(methods),
        ', '.join(amounts) or None,
        ', '.join(accts) or None,
        ', '.join(phones) or None,
        None,  # Has Media
        'YES' if strong else 'NO',
        txt,
        m['media_url'],
    ])
    threads_with_payment.add(m['thread_id'])
    mention_threads.add(m['thread_id'])
    for mm in methods: mention_method_counts[mm]+=1
    if strong: mention_strong += 1

print(f'Photo evidence rows: {len(photo_evidence)}')
print(f'Mention rows: {len(mentions_rows)}')
print(f'Threads with payment: {len(threads_with_payment)}')

# All photos in payment threads (every photo msg in those threads)
all_photos_rows = []
for m in photo_msgs:
    if m['thread_id'] not in threads_with_payment: continue
    fp = dl_by_item.get(m['item_id'])
    text = ocr.get(fp) if fp else None
    has_ocr = 'YES' if text else 'NO'
    methods, amounts, *_ = detect(text, allow_generic=True) if text else ([],[],[],[],[],[])
    time_str, _ = fmt_time(m['timestamp_us'])
    sender = m['username'] or str(m['user_id'])
    sid = str(m['user_id'])
    title = threads.get(m['thread_id'], '')
    all_photos_rows.append([
        time_str, sender, sid, title, m['thread_id'],
        'photo', has_ocr,
        ', '.join(methods) if methods else None,
        ', '.join(amounts) if amounts else None,
        m['media_url'],
    ])
print(f'All photos in payment threads rows: {len(all_photos_rows)}')

unique_senders_evidence = len(senders_payment_photos)
unique_threads_evidence = len({r[4] for r in photo_evidence})

# ===== Write Evidence workbook =====
ev_path = os.path.join(OUT, 'IG_Payment_Evidence_Detailed_2023.xlsx')
wb = xlsxwriter.Workbook(ev_path, {'constant_memory': False, 'strings_to_urls': False})
hdr = wb.add_format({'bold': True, 'bg_color':'#DDDDDD'})

# Summary
ws = wb.add_worksheet('Summary')
ws.set_column(0,0,40); ws.set_column(1,1,60)
rows = [
    ('Field','Value'),
    ('Date range','2023-04-01 to 2023-12-31'),
    ("Total photos OCR'd", total_photos_ocrd),
    ('Photos with payment evidence', len(photo_evidence)),
    ('Strong-match photos (method + amount/acct/txn)', strong_count),
    ('Unique senders involved', unique_senders_evidence),
    ('Unique threads involved', unique_threads_evidence),
    (None, None),
    ('Method', 'Photo count'),
]
for mname,_ in METHODS + [('Generic Receipt', None)]:
    if method_counts_evidence.get(mname):
        rows.append((mname, method_counts_evidence[mname]))
for i,(a,b) in enumerate(rows):
    if i==0 or (isinstance(a,str) and a=='Method'):
        ws.write(i,0,a,hdr); ws.write(i,1,b,hdr)
    else:
        ws.write(i,0,a); ws.write(i,1,b)

# Photo Payment Evidence
ws = wb.add_worksheet('Photo Payment Evidence')
widths=[20,22,14,28,22,28,20,22,18,18,18,8,80,50]
for i,w in enumerate(widths): ws.set_column(i,i,w)
headers=['Time','Sender','Sender ID','Thread Title','Thread ID','Methods Detected','Amounts','Accounts','Cards','Phones','Txn IDs','Strong','OCR Text','Media URL']
for i,h in enumerate(headers): ws.write(0,i,h,hdr)
for r,row in enumerate(photo_evidence,1):
    for c,v in enumerate(row): ws.write(r,c,v)

# All Photos in Payment Threads
ws = wb.add_worksheet('All Photos in Payment Threads')
for i,w in enumerate([20,22,14,28,22,8,14,28,20,50]): ws.set_column(i,i,w)
headers=['Time','Sender','Sender ID','Thread Title','Thread ID','Type','Has Payment OCR',"Methods (if OCR'd)","Amounts (if OCR'd)",'Media URL']
for i,h in enumerate(headers): ws.write(0,i,h,hdr)
for r,row in enumerate(all_photos_rows,1):
    for c,v in enumerate(row): ws.write(r,c,v)

# Senders Ranking
ws = wb.add_worksheet('Senders Ranking')
for i,w in enumerate([24,16,18,16,60]): ws.set_column(i,i,w)
headers=['Sender','Sender ID','Total Payment Photos','Strong Matches','Sample Amounts']
for i,h in enumerate(headers): ws.write(0,i,h,hdr)
ranked = sorted(senders_payment_photos.items(), key=lambda x:(-x[1], x[0]))
for r,(sender,n) in enumerate(ranked,1):
    ws.write(r,0,sender)
    ws.write(r,1,senders_id.get(sender,''))
    ws.write(r,2,n)
    ws.write(r,3,senders_strong.get(sender,0))
    samp = ' ; '.join(senders_amounts.get(sender, [])[:6])
    ws.write(r,4,samp or None)
wb.close()
print('Wrote', ev_path)

# ===== Write Mentions workbook =====
mn_path = os.path.join(OUT, 'IG_Payment_Mentions_2023.xlsx')
wb = xlsxwriter.Workbook(mn_path, {'constant_memory': False, 'strings_to_urls': False})
hdr = wb.add_format({'bold': True, 'bg_color':'#DDDDDD'})

ws = wb.add_worksheet('Summary')
ws.set_column(0,0,42); ws.set_column(1,1,60)
total_threads_scanned = len(threads)
total_msgs_scanned = conn.execute('SELECT COUNT(*) FROM messages').fetchone()[0]
rows = [
    ('Field','Value'),
    ('Date range','2023-04-01 to 2023-12-31'),
    ('Total threads scanned', total_threads_scanned),
    ('Total messages scanned', total_msgs_scanned),
    ('Text msgs mentioning payment (in range)', len(mentions_rows)),
    ('Strong matches (in range)', mention_strong),
    ('Unique threads with payment mentions', len(mention_threads)),
    (None, None),
    ('Method', 'Count'),
]
for mname,_ in METHODS:
    if mention_method_counts.get(mname):
        rows.append((mname, mention_method_counts[mname]))
for i,(a,b) in enumerate(rows):
    if i==0 or (isinstance(a,str) and a=='Method'):
        ws.write(i,0,a,hdr); ws.write(i,1,b,hdr)
    else:
        ws.write(i,0,a); ws.write(i,1,b)

ws = wb.add_worksheet('Payment Messages')
for i,w in enumerate([20,22,14,28,22,28,18,22,22,10,12,60,50]): ws.set_column(i,i,w)
headers=['Time','Sender','Sender ID','Thread Title','Thread ID','Methods Detected','Amounts','Account-like','Phones','Has Media','Strong Match','Message Text','Media URL']
for i,h in enumerate(headers): ws.write(0,i,h,hdr)
for r,row in enumerate(mentions_rows,1):
    for c,v in enumerate(row): ws.write(r,c,v)
wb.close()
print('Wrote', mn_path)

# Print summary
import os as _os
for p in [ev_path, mn_path]:
    print(f'{p}  size={_os.path.getsize(p)} bytes')
print(f'Evidence: Summary={len(rows)+1 if False else "n/a"}, Photo Payment Evidence rows={len(photo_evidence)}, All Photos rows={len(all_photos_rows)}, Senders Ranking={len(ranked)}')
print(f'Mentions: Payment Messages rows={len(mentions_rows)}')
