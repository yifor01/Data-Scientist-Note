Data Scientist Note
===

###### tags: `NLP` `Machine Learning`
# Topic List
:::info
- **System:** Os
- **Plot:** Matplotlib, Seaborn
- **Data Frame:** Pandas, Numpy
- **NLP:** Jieba, Seaborn
- **Crawl:** Request, Bs4, Selenium
- **Algorithm** 
:::

## System
- 執行command line
    ```python
    os.system('python main.py --input doc.txt')
    ```
- 開啟檔案視窗
    ```python
    os.startfile('doc.txt')
    ```
## Plot
- 繪圖區顏色調整(黑暗模式用)
    ```python
    plt.rc_context({'axes.edgecolor':'orange','xtick.color':'red', 
                    'ytick.color':'green', 'figure.facecolor':'white'})
    ```
- 座標label旋轉
  ```python
  plt.xticks(rotation=45)
  ```
  
- [plt中文亂碼 & 負號消失](https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/359974/)
    ```python
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    ```

- plt中文亂碼(進階Linux)
    - [simsun字體下載](http://www.font5.com.cn/zitixiazai/1/150.html)
        ```python
        from matplotlib.font_manager import FontProperties      
      font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
      ```
      
## Data Frame
* 簡體編碼 ： `GB 2312`
- 儲存csv中文亂碼
    ```python
    df.to_csv('XXXX.csv',encoding='utf_8_sig',index=False)
    df.to_excel('XXXX.xlsx',encoding='utf_8_sig',index=False)
    ```
- 特殊符號無法存檔
    ```python
    !pip install xlsxwriter
    df.to_excel("test.xlsx", engine='xlsxwriter')
    ```
- pandas 顯示更多行
    ```python
    pd.options.display.max_columns = 10
    ```
- numpy  小數點位數顯示設定
    ```python
    np.set_printoptions(precision=2)
    ```
## NLP
* [jieba詞性表](http://blog.pulipuli.info/2017/11/fasttag-identify-part-of-speech-in.html)
- 全形轉半形
    ```python
    def full2half(s):
    n = []
    for char in s:
        num = ord(char)
        if num == 0x3000:
            num = 32
        elif 0xFF01 <= num <= 0xFF5E:
            num -= 0xfee0
        num = chr(num)
        n.append(num)
    return ''.join(n)
    ```
- 清除表情符號
    ```python
    def give_emoji_free_text(text):
        allchars = [str for str in text]
        emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
        clean_text = ''.join([str for str in text if not any(i in str for i in emoji_list)])
        return clean_text
    ```
- 清除url
    ```python
    def remove_url_text(text):
        results=re.compile("(https://[a-zA-Z0-9.?/&=:]*)|(http://[a-zA-Z0-9.?/&=:]*)",re.S)
        clean_text = results.sub("",text)
        return clean_text
    ```
- 清理文字
    ```python
    def clean_txt(raw):
        fil = re.compile(r"[^0-9a-zA-Z\u4e00-\u9fa5，：？！。《》()『』「」,。【】▶%＞＜#；+-—“”:?!、<>]+", re.UNICODE)
        return fil.sub(' ', raw)
    ```
## Crawl
- 抓取代理
    ```python
    import requests
    from bs4 import BeautifulSoup
    def _get_proxies(proxy_num):
          res = requests.get('http://cn-proxy.com/archives/218')
          res.encoding = 'utf-8'
          soup = BeautifulSoup(res.text)
          soup = soup.find('div', {'class': 'entry-content'})
          proxy_ips = []
          for i in range(proxy_num):
                  proxy_ips.append(':'.join([x.text for x in soup.find('tbody').find_all('tr')[i].find_all('td')[:2]]))
          return proxy_ips
    ```
- 抓取網頁html
    ```python
    import requests
    from bs4 import BeautifulSoup
    def get_webpage(url,ip=None,cookies=None,_format='text',timeout=10):
        proxy = {'http': 'http://' + ip + '/'} if ip else None
        try:
            resp = requests.get(url=url,
                                cookies=cookies,
                                proxies=proxy,
                                timeout=timeout)
            if resp.status_code != 200:
                return None
            elif _format=='text':
                return resp.text
            elif _format=='json':
                return resp.json()        
        except:
            return None  
    ```
## Algorithm
- List search
    ```python
    from itertools import izip as zip, count
    [i for i, j in zip(count(), [‘foo’, ‘bar’, ‘baz’]) if j == ‘bar’]
    ```
