Data Scientist Note
===

###### tags: `NLP` `Machine Learning`
# Topic List
:::info
[TOC]
:::
- 文中未特別註明的code block皆為python

## System
- Python Reference
    - [Decorator](https://www.youtube.com/watch?v=5VCywjS8YEA&list=PLLuMmzMTgVK7JciUiAB8hcGA_9fQCQPlE&index=3)
- (python)執行command line
    ```python
    os.system('python main.py --input doc.txt')
    ```
- (python)開啟檔案視窗
    ```python
    os.startfile('doc.txt')
    ```
- (command)UnicodeEncodeError
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
- plt中文亂碼(Linux簡易版)
    - 指定font位置[(simsun字體下載)](http://www.font5.com.cn/zitixiazai/1/150.html)
        ```python
        from matplotlib.font_manager import FontProperties      
      font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
      ```
- plt中文亂碼(Linux進階版)
    - (python)查找matplotlib路徑 
  ```python
    import matplotlib
    print(f'matplotlib system path: {matplotlib.__file__}')
  ```
    - (command line)將font檔案丟到上面的目錄+"/mpl-data/fonts/ttf"
    ```os
    cd [matplotlib path]/mpl-data/fonts/ttf
    ```
    - (command line)更改資料權限
    ```os
    chmod 755 [font file]
    ```
    - (command line)刪除matplotlib快取檔案
    ```os
    cd ~/.cache/matplotlib
    rm -rf *.*
    ```
    - (python)查看字型安裝狀況
    ```python
    import matplotlib.font_manager
    ttf_list = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

    for _ttf in ttf_list:
        print(_ttf)
    ```
    - (python)載入字型
    ```python
    import matplotlib
    matplotlib.rcParams[u'font.sans-serif'] = ['Taipei Sans TC Beta']
    matplotlib.rcParams['axes.unicode_minus'] = False
    ```
    - (python)測試
    ```python
    plt.plot([1,2,3],[1,2,3])
    plt.title('測試')
    ```
    



## Data Frame
* 簡體編碼 ： `GB 2312`
- 儲存csv中文亂碼
    ```python
    # 僅read csv 可用dtype={'A':str}
    df.to_csv('XXXX.csv',encoding='utf_8_sig',index=False)
    df.to_excel('XXXX.xlsx',encoding='utf_8_sig',index=False)
    ```
- url 過多導致excel的錯誤 & 檔案切分 & 壓縮多個檔案
    ```python 
    import pandas as pd
    import numpy as np
    max_output, output_file_name = 1000000, 'testdata'
    batch_size = int(np.ceil(len(df)/max_output))
    for batch in batch_size:
        writer = pd.ExcelWriter(f'{output_file_name}_part{batch+1}.xlsx', 
                                engine='xlsxwriter',
                                options={'strings_to_urls': False})
        df.iloc[(batch)*max_output:(batch+1)*max_output].to_excel(writer, sheet_name='Sheet1')
        writer.save()
        
    import zipfile,os
    from glob import glob
    with zipfile.ZipFile(f'{output_file_name}_all.zip', 'w') as zf:
        for file in glob(f'{output_file_name}_part*.xlsx'):
            zf.write(file)
            #os.remove(file)
    ```
- excel 儲存多個sheet
    ```python
    writer = pd.ExcelWriter('df.xlsx', engine = 'xlsxwriter')
    df1.to_excel(writer, sheet_name = 'sheet1',index=0)
    df2.to_excel(writer, sheet_name = 'sheet2',index=0)
    writer.save()
    writer.close()
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
* [NLP repo](https://notebooks.quantumstat.com/?fbclid=IwAR09na0S8ZFIEI6hQASptNAsw29EP6tmkZAiqH-Y6P25487OA9EXOsh2NVM)
* [PyTorch cheat sheet](https://hackmd.io/@rh0jTfFDTO6SteMDq91tgg/HkDRHKLrU)
* [PyTorch 中文教程](https://pytorch.apachecn.org/docs/1.2/)
* [Jieba詞性表](http://blog.pulipuli.info/2017/11/fasttag-identify-part-of-speech-in.html)
* [MONPA 罔拍繁體中文斷詞](https://github.com/monpa-team/monpa)
* [中文任務benchmarks：CLUE](https://www.cluebenchmarks.com/rank.html)
* [中文任務BERT Pre-Train Model](https://github.com/ymcui/Chinese-BERT-wwm)
* [BERT colab](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb
) (不適用tf2)
* [同義詞Wordnet](https://blog.csdn.net/Pursue_MyHeart/article/details/80631278)
* [正規表達式教學](https://cloud.tencent.com/developer/article/1597800)
* [UTF8編碼](https://blog.miniasp.com/post/2019/01/02/Common-Regex-patterns-for-Unicode-characters)

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
        return results.sub("",text)
    ```
- 清理文字
    ```python
    def clean_txt(raw):
        fil = re.compile(r"[^0-9a-zA-Z\u4E00-\u9FFF，：？！。《》()『』「」,。【】▶%＞＜#；+-—“”:?!、<>]+", re.UNICODE)
        return fil.sub(' ', raw)
    ```
#### Modeling
- BERT output attention： [Layer(12)][multi-head(12)][sent-dim][sent-dim]


## Crawling
- 抓取代理伺服器 (以 http://cn-proxy.com/archives/218 為例)
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
    [i for i, j in zip(count(), ['foo', 'bar', 'baz']) if j == 'bar']
    ```
- str2list
    ```python
    txt_list = u"['理財', '基金', '量化', '交易']"
    eval(txt_list) # ['理財', '基金', '量化', '交易']
    ```


## Docker
- (command)build tf enviroment with GPU and Jupyter
    ```docker
    docker run --runtime=nvidia -d -p 27390:8888 -v "$(pwd)"/myfile:/tf --name tf_rnn tensorflow/tensorflow:latest-gpu-py3-jupyter 
    docker exec tf_rnn jupyter notebook list # get jupyter token
    ```
- (command)build pytorch enviroment with GPU
    ```docker
    docker run --runtime=nvidia -it -p 12222:8888 -p 16666:6006 -v "$(pwd)"/torch:/workspace --name howard_torch pytorch/pytorch:1.0-cuda10.0-cudnn7-devel 
    ```
- (command)docker install jupyter 
    ```docker
    pip install --user jupyter && pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install
    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root 
    ```
