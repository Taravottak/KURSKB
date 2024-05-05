import requests
import json
def instructions():
    '''
        перед работой кода:
        pip install requests
        если не установится два пути через команд строку или через пичарм интерпритатор
        о функциях:
        get_into - #вся база данных в json с выводом в текст файл
        get_ticker - #метод получения иинформации о паре или парах за последние 24 часа
        get_depth - #метод возврата инфо о выставленных на продажу и покупку ордерах
        get_trades - #совершенные сделки по покупке и продаже
        #total_bids_amount - общая сумма закупа последних 150 ордеров
        #limit=150 глубина вывод по умолч, а макс 2000
        #response = requests.get(url="https://yobit.net/api/3/ticker/eth_btc-xrp_btc?ignore_invalid=1")
        #?ignore_invalid=1 - игнорирует несущ пары
        #asks - словарь с ордерами на продажу
        #bits- словарь с ордерами на покупку
        #значения словарей списки первое знаечение прайс второе кол-во монет
        #ЗАМЕЧАНИЕ-тк курс монеты обновляется с каким-то промежутком времени, чтобы проверить правильность данных\
        код надо запускать заново и вставлять json-файл в json viewer тем самым проверяя выведенный массив данных
        if (os.path.exists('depth_bids.txt'))==False:
        '''
def get_into():
    response = requests.get(url="https://yobit.net/api/3/info")
    with open("into.txt", "w") as file:
        file.write(response.text)
    return response.text

def get_depth_sell(coin1="btc",coin2="usdt", limit=150):
    response = requests.get(url=f"https://yobit.net/api/3/depth/{coin1}_{coin2}?limit={limit}&ignore_invalid=1")

    asks = response.json()[f"{coin1}_{coin2}"]["asks"]
    with open("depth_asks.txt", "w") as file:
        for ask in asks:
            file.write(f"{(ask[0])} {(ask[1]):.6f}\n")

    total_asks_amount=0
    sell=[]
    for item in asks:
        price=item[0]
        coin_amount=item[1]
        total_asks_amount+=price*coin_amount
        sell.append(price * coin_amount)

    with open("depth_sell.txt", "w") as file:
        for i in range(limit): #698.5997120000001/6435.783671918644
            file.write(f"{(sell[i])}\n")

    return f"Total Asks: {total_asks_amount} $",sell

def get_depth_buy(coin1="btc",coin2="usdt", limit=150):
    response = requests.get(url=f"https://yobit.net/api/3/depth/{coin1}_{coin2}?limit={limit}&ignore_invalid=1")

    bids = response.json()[f"{coin1}_{coin2}"]["bids"]
    with open("depth_bids.txt", "w") as file:
        for bid in bids:
            file.write(f"{(bid[0])} {(bid[1]):.6f}\n")

    total_bids_amount=0
    buy=[]
    for item in bids:
        price=item[0]
        coin_amount=item[1]
        total_bids_amount+=price*coin_amount
        buy.append(price*coin_amount)

    with open("depth_buy.txt", "w") as file:
        for i in range(limit): #698.5997120000001/6435.783671918644
            file.write(f"{(buy[i])}\n")

    return f"Total Bids: {total_bids_amount} $",buy
def main():
    #print(get_into())
    #print(get_depth())
    #print(get_depth_buy(coin1="btc",limit=30))
    print(get_depth_sell(coin1="btc", limit=30))


if __name__ == "__main__":
    main()