Data Scientist Note
===

###### tags: `NLP` `Machine Learning`
- **System:** Os
- **Plot:** Matplotlib, Seaborn
- **Data Frame:** Pandas, Numpy
- **NLP:** Jieba, Seaborn
- **Crawl:** Request, Bs4, Selenium
- **Algorithm** 

## System
- 執行command line
      <pre><code>os.system('python main.py --input doc.txt')
</code></pre>
- 開啟檔案視窗
      <pre><code>os.startfile('doc.txt')</code></pre>

## Plot

- 繪圖區顏色調整(黑暗模式用)
      <pre><code>plt.rc_context({'axes.edgecolor':'orange', 'xtick.color':'red', 
      'ytick.color':'green', 'figure.facecolor':'white'})</code></pre>
- 座標label旋轉
      <pre><code>plt.xticks(rotation=45)</code></pre>
- [plt中文亂碼 & 負號消失](https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/359974/)
      <pre><code>plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False</code></pre>
- plt中文亂碼(進階Linux)
    - [simsun字體下載](http://www.font5.com.cn/zitixiazai/1/150.html)
    <pre><code>from matplotlib.font_manager import FontProperties      
  font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)</code></pre>
      
## Data Frame
* 簡體編碼 ： `GB 2312`
- 儲存csv中文亂碼
      <pre><code>df.to_csv('XXXX.csv',encoding='utf_8_sig',index=False)
      df.to_excel('XXXX.xlsx',encoding='utf_8_sig',index=False)</code></pre>
- 特殊符號無法存檔
      <pre><code>!pip install xlsxwriter
      df.to_excel("XXXX.xlsx", engine='xlsxwriter')</code></pre>
- pandas 顯示更多行
      <pre><code>pd.options.display.max_columns = 10
</code></pre>
- numpy  小數點位數顯示設定
      <pre><code>np.set_printoptions(precision=2)
</code></pre>

## NLP
* [jieba詞性表](http://blog.pulipuli.info/2017/11/fasttag-identify-part-of-speech-in.html)
- 全形轉半形
      <pre><code>def full2half(s):
    &nbsp;&nbsp;&nbsp;&nbsp;n = []
    &nbsp;&nbsp;&nbsp;&nbsp;for char in s:
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;num = ord(char)
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if num == 0x3000:
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;num = 32
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;elif 0xFF01 <= num <= 0xFF5E:
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;num -= 0xfee0
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;num = chr(num)
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;n.append(num)
    &nbsp;&nbsp;&nbsp;&nbsp;return ''.join(n)
</code></pre>
- 清除表情符號
      <pre><code>import emoji
      def give_emoji_free_text(text):
      &nbsp;&nbsp;&nbsp;&nbsp;allchars = [str for str in text]
    &nbsp;&nbsp;&nbsp;&nbsp;emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
    &nbsp;&nbsp;&nbsp;&nbsp;clean_text = ''.join([str for str in text if not any(i in str for i in emoji_list)])
    &nbsp;&nbsp;&nbsp;&nbsp;return clean_text</code></pre>
- 清除url
      <pre><code>import re
      def remove_url_text(text):
&nbsp;&nbsp;&nbsp;&nbsp;results = re.compile("(https://[a-zA-Z0-9.?/&=:]*)|(http://[a-zA-Z0-9.?/&=:]*)", re.S)
    &nbsp;&nbsp;&nbsp;&nbsp;clean_text = results.sub("", text)
    &nbsp;&nbsp;&nbsp;&nbsp;return clean_text</code></pre>
- 清理文字
      <pre><code>import re
      def cleantxt(raw):
    &nbsp;&nbsp;&nbsp;&nbsp;fil = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5，：？！。《》()『』「」,。【】▶%＞＜#；+-—“”:?!、<>]+", re.UNICODE)
    &nbsp;&nbsp;&nbsp;&nbsp;return fil.sub(' ', raw)</code></pre>



## Crawl
- 抓取代理
    - [全球代理列表](http://cn-proxy.com/archives/218)
     <pre><code>import requests
  from bs4 import BeautifulSoup
  def _get_proxies(proxy_num):
    &nbsp;&nbsp;&nbsp;&nbsp;res = requests.get('http://cn-proxy.com/archives/218')
    &nbsp;&nbsp;&nbsp;&nbsp;res.encoding = 'utf-8'
    &nbsp;&nbsp;&nbsp;&nbsp;soup = BeautifulSoup(res.text)
    &nbsp;&nbsp;&nbsp;&nbsp;soup = soup.find('div', {'class': 'entry-content'})
    &nbsp;&nbsp;&nbsp;&nbsp;proxy_ips = []
    &nbsp;&nbsp;&nbsp;&nbsp;for i in range(proxy_num):
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;proxy_ips.append(':'.join([x.text for x in soup.find('tbody').find_all('tr')[i].find_all('td')[:2]]))
    &nbsp;&nbsp;&nbsp;&nbsp;return proxy_ips</code></pre>
- 抓取網頁html
      <pre><code>import requests
from bs4 import BeautifulSoup
def get_webpage(url,ip=None,cookies=None,_format='text',timeout=10):
&nbsp;&nbsp;&nbsp;&nbsp;# 輸入代理伺服器 & 模擬網頁請求
    &nbsp;&nbsp;&nbsp;&nbsp;proxy = {'http': 'http://' + ip + '/'} if ip else None
    &nbsp;&nbsp;&nbsp;&nbsp;headers = {'User-Agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/81.0.4044.113 Safari/537.36'}
    &nbsp;&nbsp;&nbsp;&nbsp;try:
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;resp = requests.get(url=url,cookies=cookies,proxies=proxy,headers=headers,timeout=timeout)
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;if resp.status_code != 200:
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;print(f'Connect error: {resp.status_code}')
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return None
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;elif _format=='text':
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return resp.text
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;elif _format=='json':
            &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return resp.json()        
    &nbsp;&nbsp;&nbsp;&nbsp;except:
        &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;return None  
</code></pre>







## Algorithm
- List search
      <pre><code>from itertools import izip as zip, count
      [i for i, j in zip(count(), ['foo', 'bar', 'baz']) if j == 'bar']</code></pre>










<!--:mag: Sprint Retro
---
##################################################################################################
### What we can start Doing
- New initiatives and experiments we want to start improving



:closed_book: Tasks
--
==Importance== (1 - 5) / Name / **Estimate** (1, 2, 3, 5, 8, 13)
### Development Team:
- [ ] ==5== Email invite
  - [x] ==4== Email registration page **5**
  - [ ] ==5== Email invitees **3**
- [ ] ==4== Setup e2e test in production **2**-->
