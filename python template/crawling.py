import requests
from bs4 import BeautifulSoup
import pandas as pd
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
plt.rcParams['axes.unicode_minus'] = False


def main(show_plot=True):
    print('=== 蘋果今日焦點 ===')
    # Use request to get html (for general)
    dom = requests.get(
        'http://www.appledaily.com.tw/appledaily/hotdaily/headline')
    try:
        if dom.status_code == 200:
            soup = BeautifulSoup(dom.text, 'html5lib')
            for i, ele in enumerate(soup.find('ul', 'all').find_all('li')):
                print(
                    f'{i+1:2d} :',
                    ele.find('div', 'aht_title').text,
                )
        else:
            print(f'Connect error code: {dom.status_code}')
    except BaseException:
        print('Connect timeout!')

    # Use pandas to get html (for dataframe)
    print('=== 永和明天天氣 ===')
    dom = pd.read_html(
        'https://www.cwb.gov.tw/V8/C/W/Town/MOD/3hr/6500400_3hr_PC.html?T=2020033020-0',
        encoding='utf-8')[0]
    dom = dom.drop(index=[0, 1, 3, 5, 7, 9, 10, 11, 12])
    dom.columns = [x[-8:] + x[:-8] for x in dom.columns]
    dom = dom.T.reset_index()
    dom.columns = ['日期', '溫度', '降雨機率', '體感溫度', '相對溼度']
    dom['溫度'] = dom['溫度'].astype('int') / 100
    dom['體感溫度'] = dom['體感溫度'].astype('int') / 100
    dom['降雨機率'] = dom['降雨機率'].apply(lambda x: float(x[:-1]) / 100)
    dom['相對溼度'] = dom['相對溼度'].apply(lambda x: float(x[:-1]) / 100)
    display(dom)

    if show_plot:
        fig = plt.figure()
        ax1 = fig.add_subplot(111)
        ax1.plot('日期', '溫度', data=dom, label='溫度', color='r')
        ax1.set_title("永和未來天氣")
        for tick in [x for x in ax1.get_xticklabels()]:
            tick.set_rotation(45)
        ax2 = ax1.twinx()
        ax2.plot('日期', '降雨機率', data=dom, label='降雨機率', linestyle='dashed')
        for tick in [x for x in ax2.get_xticklabels()]:
            tick.set_rotation(45)
        ax1.legend(
            bbox_to_anchor=(
                0.01,
                0.2),
            loc='upper left',
            borderaxespad=0)
        ax2.legend(
            bbox_to_anchor=(
                0.01,
                0.1),
            loc='upper left',
            borderaxespad=0)
        ax2.set_ylim([0, 1.05])
        plt.show()


if __name__ == '__main__':
    main()
