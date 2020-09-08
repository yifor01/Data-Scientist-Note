Data Scientist Note
===

###### tags: `NLP` `Machine Learning`
# Topic List
:::info
[TOC]
:::
- Default coding language: `python`

## System
- Python Reference
    - [Decorator](https://www.youtube.com/watch?v=5VCywjS8YEA&list=PLLuMmzMTgVK7JciUiAB8hcGA_9fQCQPlE&index=3)
- (python) Execute command line
    ```python
    os.system('python main.py --input doc.txt')
    ```
- (python) Open the file wondow.
    ```python
    os.startfile('doc.txt')
    ```
- (command) UnicodeEncodeError
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
- Drawing area color adjustment (for Dark Mode)
    ```python
     plt.rc_context({'axes.edgecolor':'orange','xtick.color':'red', 
                        'ytick.color':'green', 'figure.facecolor':'white'})
    ```
- Label rotation
  ```python
  plt.xticks(rotation=45)
  ```
- [`plt` package chinese font random code & minus sign disappear](https://codertw.com/%E7%A8%8B%E5%BC%8F%E8%AA%9E%E8%A8%80/359974/)
    ```python
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
    plt.rcParams['axes.unicode_minus'] = False
    ```
- `plt` package chinese random code (for Linux)
    - Specify font path[(simsun font download)](http://www.font5.com.cn/zitixiazai/1/150.html)
        ```python
        from matplotlib.font_manager import FontProperties      
      font = FontProperties(fname=r"c:\windows\fonts\simsun.ttc", size=14)
      ```
- `plt` package chinese font random code (Linux advanced)
    - (python) Find matplotlib path 
  ```python
    import matplotlib
    print(f'matplotlib system path: {matplotlib.__file__}')
  ```
    - (command line) Drop the font file into the above directory and add "/mpl-data/fonts/ttf"
        ```os
        cd [matplotlib path]/mpl-data/fonts/ttf
        ```
    - (command line) Change data permission
        ```os
        chmod 755 [font file]
        ```
    - (command line) Delete matplotlib cache file
        ```os
        cd ~/.cache/matplotlib
        rm -rf *.*
        ```
    - (python) Check font install
        ```python
        import matplotlib.font_manager
        ttf_list = sorted([f.name for f in matplotlib.font_manager.fontManager.ttflist])

        for _ttf in ttf_list:
            print(_ttf)
        ```
    - (python) Load font
        ```python
        import matplotlib
        matplotlib.rcParams[u'font.sans-serif'] = ['Taipei Sans TC Beta']
        matplotlib.rcParams['axes.unicode_minus'] = False
        ```
    - (python) Test
        ```python
        plt.plot([1,2,3],[1,2,3])
        plt.title('測試')
        ```




## Data Frame
* Simplified Chinese coding ： `GB 2312`
- Save Chinese font random code in `csv` file
    ```python
    # only `read csv` can use `dtype={'A':str}`
    df.to_csv('XXXX.csv',encoding='utf_8_sig',index=False)
    df.to_excel('XXXX.xlsx',encoding='utf_8_sig',index=False)
    ```
- Too many URLs causing excel errors & file spliting & compress multiple files
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
- `excel` save more than one sheet
    ```python
    writer = pd.ExcelWriter('df.xlsx', engine = 'xlsxwriter')
    df1.to_excel(writer, sheet_name = 'sheet1',index=0)
    df2.to_excel(writer, sheet_name = 'sheet2',index=0)
    writer.save()
    writer.close()
    ```
- Special symbols cannot be archived error
    ```python
    !pip install xlsxwriter
    df.to_excel("test.xlsx", engine='xlsxwriter')
    ```
- `pandas` show more columns
    ```python
    pd.options.display.max_columns = 10
    ```
- `numpy` decimal Point Display Settings
    ```python
    np.set_printoptions(precision=2)
    ```
#### Tricks
- `pandas` sampleing (without replace)
    ```python
    # Set `frac=1` meaning data reorder
    df.sample(frac=0.8).reset_index(drop=True) 
    ```  
- `pandas` reduce memory usage
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
- Text Segmentation
    - [`Jieba`: part of speech table](http://blog.pulipuli.info/2017/11/fasttag-identify-part-of-speech-in.html)
    - [`MONPA`: traditional chinese text segmentation system](https://github.com/monpa-team/monpa)

- DL model demo 
    - [**NLP repo**](https://notebooks.quantumstat.com/?fbclid=IwAR09na0S8ZFIEI6hQASptNAsw29EP6tmkZAiqH-Y6P25487OA9EXOsh2NVM) 
    - [**NLP benchmark model demo**](https://models.quantumstat.com/?fbclid=IwAR0BR6kgG-fEIvARNZtvj3QnutJopeU1Dt9Gkk6eaKGVH2q1NAmI2S_9YNo) 
    - [**SOTA model list**](https://paperswithcode.com/sota) 
    - [Chinese task benchmarks：CLUE](https://www.cluebenchmarks.com/rank.html)
    - [BERT colab](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb
) (no support tf2)

- Pytorch
    - [PyTorch cheat sheet](https://hackmd.io/@rh0jTfFDTO6SteMDq91tgg/HkDRHKLrU)
    - [PyTorch Chinese tutorial](https://pytorch.apachecn.org/docs/1.2/)
    - [Optimize Pytorch by using Optuna](https://medium.com/pytorch/using-optuna-to-optimize-pytorch-ignite-hyperparameters-626ffe6d4783)

- DL model reference
    - [**hugging face : pre-trained model library**](https://huggingface.co/models?filter=pytorch) 
    - [BERT Chinese Pre-Train Model](https://github.com/ymcui/Chinese-BERT-wwm)
    - [Speed up BERT](https://medium.com/pytorch/using-optuna-to-optimize-pytorch-ignite-hyperparameters-626ffe6d4783)
    - [Fine tuning BERT with TF](https://medium.com/serendeepia/finetuning-bert-with-tensorflow-estimators-in-only-a-few-lines-of-code-f522dfa2295c)

- others
    * [Synonym Wordnet](https://blog.csdn.net/Pursue_MyHeart/article/details/80631278)
    * [Regular expression tutorial ](https://cloud.tencent.com/developer/article/1597800)
    * [UTF8 encoding](https://blog.miniasp.com/post/2019/01/02/Common-Regex-patterns-for-Unicode-characters)


#### Data Processing
- Full text to half text
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
- Clean emoji 
    ```python
    import emoji
    def give_emoji_free_text(text):
        allchars = [str for str in text]
        emoji_list = [c for c in allchars if c in emoji.UNICODE_EMOJI]
        clean_text = ''.join([str for str in text if not any(i in str for i in emoji_list)])
        return clean_text
    ```
- emoji to text (part1)
    ```python
    def emoji_cleaning(text):
        # Change emoji to text
        text = emoji.demojize(text).replace(":", " ")

        # Delete repeated emoji
        tokenizer = text.split()
        repeated_list = []
        for word in tokenizer:
            if word not in repeated_list:
                repeated_list.append(word)

        text = ' '.join(text for text in repeated_list)
        text = text.replace("_", " ").replace("-", " ")
        return text
    ```
- emoji to text (part2)
    ```python
    def text_emoji_transform(text):
        text = text.lower()
        text = re.sub(r'\n', '', text)
        
        # change emoticon to text
        text = re.sub(r':\(', 'dislike', text)
        text = re.sub(r': \(\(', 'dislike', text)
        text = re.sub(r':, \(', 'dislike', text)
        text = re.sub(r':\)', 'smile', text)
        text = re.sub(r';\)', 'smile', text)
        text = re.sub(r':\)\)\)', 'smile', text)
        text = re.sub(r':\)\)\)\)\)\)', 'smile', text)
        text = re.sub(r'=\)\)\)\)', 'smile', text)
        tokenizer = text.split()
        return ' '.join([text for text in tokenizer])
    ```
- Clear English repeated characters (ex: "likkkkkkkkke")
    ```python
    def delete_repeated_char(text):
        text = re.sub(r'(\w)\1{2,}', r'\1', text)
        return text
    ```
- Clean url
    ```python
    def remove_url_text(text):
        results=re.compile("(https://[a-zA-Z0-9.?/&=:]*)|(http://[a-zA-Z0-9.?/&=:]*)",re.S)
        return results.sub("",text)
    ```
- Clean text
    ```python
    def clean_txt(raw):
        fil = re.compile(r"[^0-9a-zA-Z\u4E00-\u9FFF，：？！。《》()『』「」,。【】▶%＞＜#；+-—“”:?!、<>]+", re.UNICODE)
        return fil.sub(' ', raw)
    ```
#### Modeling
- BERT output attention： [Layer(12)][multi-head(12)][sent-dim][sent-dim]


## Crawling
- Crawling proxy servers ( http://cn-proxy.com/archives/218 as exmaple)
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
- Crawling website html
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
- (command) build tf enviroment with GPU and Jupyter
    ```docker
    docker run --runtime=nvidia -d -p 27390:8888 -v "$(pwd)"/myfile:/tf --name tf_rnn tensorflow/tensorflow:latest-gpu-py3-jupyter 
    docker exec tf_rnn jupyter notebook list # get jupyter token
    ```
- (command) build pytorch enviroment with GPU
    ```docker
    docker run --runtime=nvidia -it -p 12222:8888 -p 16666:6006 -v "$(pwd)"/torch:/workspace --name howard_torch pytorch/pytorch:1.0-cuda10.0-cudnn7-devel 
    ```
- (command) docker install jupyter 
    ```docker
    pip install --user jupyter && pip install jupyter_contrib_nbextensions && jupyter contrib nbextension install
    jupyter notebook --ip 0.0.0.0 --no-browser --allow-root 
    ```
