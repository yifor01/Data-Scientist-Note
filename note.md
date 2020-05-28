Data Scientist Note
===

###### tags: `NLP` `Machine Learning`
# Topic List
:::info
[TOC]
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
- UnicodeEncodeError
    ```python
    PYTHONIOENCODING=utf-8 python [python file]
    ```
- autopep8 in place
    - PyCharm External Tools: 
        - Programs: autopep8 (C:\Users\User\Anaconda3\Scripts\autopep8.exe)
        - Parameters: --in-place --aggressive --aggressive \$FilePath\$
        - Working directory: \$ProjectFileDir\$
        - Output Files: \$FILE_PATH\$\\:\$LINE\$\\:$COLUMN\$\\:.*
    - Command Line 
        ```python
        autopep8 --in-place --aggressive --aggressive <filename>
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
    # 僅csv 可用dtype={'A':str}
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
#### Tricks
- pandas 抽樣 (without replace)
    ```python
    # 設定frac=1為重排序
    df.sample(frac=0.8).reset_index(drop=True) 
    ```  
- pandas 降低記憶體使用量
    ```python
    def reduce_mem_usage(df, verbose=True):
        numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
        start_mem = df.memory_usage().sum() / 1024**2    
        for col in df.columns:
            col_type = df[col].dtypes
            if col_type in numerics:
                c_min = df[col].min()
                c_max = df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        df[col] = df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        df[col] = df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        df[col] = df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        df[col] = df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        df[col] = df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        df[col] = df[col].astype(np.float32)
                    else:
                        df[col] = df[col].astype(np.float64)    
        end_mem = df.memory_usage().sum() / 1024**2
        if verbose: print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(end_mem, 100 * (start_mem - end_mem) / start_mem))
        return df
    ```  



## NLP
#### NLP Reference
* [PyTorch cheat sheet](https://hackmd.io/@rh0jTfFDTO6SteMDq91tgg/HkDRHKLrU)
* [PyTorch 中文教程](https://pytorch.apachecn.org/docs/1.2/)
* [Jieba詞性表](http://blog.pulipuli.info/2017/11/fasttag-identify-part-of-speech-in.html)
* [MONPA 罔拍繁體中文斷詞](https://github.com/monpa-team/monpa)
* [中文任務benchmarks：CLUE](https://www.cluebenchmarks.com/rank.html)
* [中文任務BERT Pre-Train Model](https://github.com/ymcui/Chinese-BERT-wwm)
* [BERT colab](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb
) (不適用tf2)



#### Data Processing
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
    import emoji
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
## Crawling
- 抓取代理伺服器
    ```python
    import requests
    from bs4 import BeautifulSoup
    def _get_proxies(proxy_num=10):
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
- Multiple "or" conditions
    ```python
    x = 10
    if x > 5 or x < 9:
        print('x in range')
    if not 5 < x < 9:
        print('x not in range')
    if any([x > 5, x < 9]):
        print('x in range')
    ```
- Create dictionary
    ```python
    keys = ['AJ1', '椰子鞋', 'Boost 350']
    values = ['NIKE', 'ADIDAS', 'ADIDAS']
    d = dict(zip(keys, values))
    print(d) # {'AJ1': 'NIKE', '椰子鞋': 'ADIDAS', 'Boost V2': 'ADIDAS'}
    ```
- List search
    ```python
    from itertools import izip as zip, count
    [i for i, j in zip(count(), [‘foo’, ‘bar’, ‘baz’]) if j == ‘bar’]
    ```


## Docker
- build tf enviroment with GPU and Jupyter
    ```docker
    docker run --runtime=nvidia -d -p 27390:8888 -v "$(pwd)"/myfile:/tf --name tf_rnn tensorflow/tensorflow:latest-gpu-py3-jupyter 
    docker exec tf_rnn jupyter notebook list # get jupyter token
    ```
- build pytorch enviroment with GPU
    ```docker
    docker run --runtime=nvidia -it -p 12222:8888 -p 16666:6006 -v "$(pwd)"/torch:/workspace --name howard_torch pytorch/pytorch:1.0-cuda10.0-cudnn7-devel 
    ```
    
    
