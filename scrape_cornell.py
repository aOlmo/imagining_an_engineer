from selenium import webdriver
import time
import urllib
import urllib2
import os
import requests

def main():
    os.chdir('script')
    browser = webdriver.Firefox()
    search_url = 'https://www.engineering.cornell.edu/faculty-directory?letter='

    
    counter = 1
    
    
    for i in range(65, 91):
        browser.get(search_url + chr(i))
        time.sleep(1)
        
        images = browser.find_elements_by_tag_name('img')
        image_list = []
        for img in images:
            image_list.append(img.get_attribute('src'))
        
        image_list = image_list[2:]
        for image in image_list:
            print image
            file_extension = image.split(".")[-1].split("?")[0]
            download_image(image, "prof_corn" + str(counter) + "." + file_extension)
            counter = counter+1

    browser.quit()

def download_image(url, filename):
    #print url
    req = urllib2.Request(url, headers={'User-Agent' : "Magic Browser"})
    filedata = urllib2.urlopen(req)
    filedata = urllib2.urlopen(url)
    datatowrite = filedata.read()
    with open(filename, 'wb') as f:
        f.write(datatowrite)
    #print con.read()
    '''url = url.strip("http://")
        url = "http://"+ url
        if 'tamu' not in url:
        url = url[:7] + 'mays.tamu.edu' + url[7:]
        print "url", url
        img_data = requests.get(url).content
        with open (filename, 'wb') as handler:
        handler.write(img_data)'''
#urllib.urlretrieve(url, filename)

main()


