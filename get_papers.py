# _*_ coding:utf-8 _*_
from selenium import webdriver
import time


def get_papers():
    driver = webdriver.Chrome('./chromedriver')
    driver.get('https://iclr.cc/Conferences/2018/Schedule')
    time.sleep(10)
    count = len(driver.find_elements_by_xpath(
            "//div[@class='col-xs-12 col-sm-9']/div"))
    # count = 6  for debug
    papers = []
    for i in range(3, count+1):
        xp = "//div[@class='col-xs-12 col-sm-9']/div["+str(i)+"]/div"
        driver.find_elements_by_xpath(xp)[0].click()
        time.sleep(3)
        title = driver.find_elements_by_xpath(
                 "//div[@class='maincardBody']")[0].text
        abstract = driver.find_elements_by_xpath(
                   "//div[@class='abstractContainer']")[0].text
        url = driver.current_url
        papers.append([str(i-2), title, abstract, url])
        driver.back()
        time.sleep(3)
    driver.quit()
    return papers  # papers =[[id,title,abstract,html],]
