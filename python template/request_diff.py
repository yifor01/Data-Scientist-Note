import time
import asyncio
import requests

'''
比較直接request與透過asyncio request速度差異
'''

def now(): return time.time()


loop = asyncio.get_event_loop()
test_url = r'http://share.dmhy.org/topics/list'
MAX_PAGE = 30

# method 1
def get_web_page(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/81.0.4044.113 Safari/537.36', }
    resp = requests.get(url=url, headers=headers)
    if resp.status_code != 200:
        print('Invalid url:', resp.url)
        return None
    else:
        return resp.text

# method 2
async def send_req(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) '
                      'Chrome/81.0.4044.113 Safari/537.36', }
    res = await loop.run_in_executor(None, requests.get, url, headers)
    res.encoding = 'urf-8'
    result = res.text
    return result


# method 1
start = now()
res1 = []
for page in range(1, MAX_PAGE):
    _res = get_web_page(f'{test_url}/{page}')
    res1.append(_res)
print(f'Method 1 using time: {now() - start}')  # 28s

# method 2
start = now()
loop = asyncio.get_event_loop()
tasks = []
for page in range(1, MAX_PAGE):
    url = f'/{page}'
    task = loop.create_task(send_req(f'{test_url}/{page}'))
    tasks.append(task)
loop.run_until_complete(asyncio.wait(tasks))
print(f'Method 2 using time: {now() - start}')  # 3s
