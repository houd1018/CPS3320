from selenium import webdriver
from selenium.webdriver.common.by import By
import requests
import io
from PIL import Image
import time

PATH = "C:\\Users\\Howard\\Desktop\\CPS HW\\CPS3320\\project\\image scrape\\chromedriver.exe"

wd = webdriver.Chrome(PATH)

def get_images_from_google(wd, delay, max_images):
	def scroll_down(wd):
		# JS executable script
		wd.execute_script("window.scrollTo(0, document.body.scrollHeight);")
		# give delay to load the remaining of the images
		time.sleep(delay)

	# google image web page that you wanna scrape
	url = "https://www.google.com/search?q=cat&tbm=isch&ved=2ahUKEwjt6uve47H3AhUkGVkFHYKND_sQ2-cCegQIABAA&oq=cat&gs_lcp=CgNpbWcQAzIHCAAQsQMQQzIKCAAQsQMQgwEQQzIHCAAQsQMQQzIECAAQQzIECAAQQzIICAAQgAQQsQMyBAgAEEMyBwgAELEDEEMyCggAELEDEIMBEEMyBAgAEENQ_gdYyQlgjw1oAHAAeACAAYwEiAGaC5IBBzMtMy4wLjGYAQCgAQGqAQtnd3Mtd2l6LWltZ8ABAQ&sclient=img&ei=MuxnYu2JNKSy5NoPgpu-2A8&bih=761&biw=1550"
	# load this page with webdriver
	wd.get(url)

	# make sure we do not have duplicate url
	image_urls = set()
	skips = 0

	while len(image_urls) + skips < max_images:

		# scroll down to the bottom
		scroll_down(wd)

		# put thumbnails' class name here
		# By is used to specify we are looking for class name
		thumbnails = wd.find_elements(By.CLASS_NAME, "Q4LuWd")

		# adding new images
		for img in thumbnails[len(image_urls) + skips:max_images]:
			try:
				# click the thumbnail
				img.click()
				# give time to poped up window
				time.sleep(delay)
			except:
				continue
			
			# find multiple larger images
			# specify your source image's class name
			images = wd.find_elements(By.CLASS_NAME, "n3VNCb")
			for image in images:
				
				# skip images are existent in image_urls
				if image.get_attribute('src') in image_urls:
					max_images += 1
					skips += 1
					break

				# check whether it has a 'src' tag
				# make sure we get a valid link
				if image.get_attribute('src') and 'http' in image.get_attribute('src'):
					image_urls.add(image.get_attribute('src'))
					print(f"Found {len(image_urls)}")

	return image_urls

def download_image(download_path, url, file_name):
	try:
		# ger content by request
		image_content = requests.get(url).content

		# transform to image	
		image_file = io.BytesIO(image_content)
		image = Image.open(image_file)
		file_path = download_path + file_name

		# save image
		with open(file_path, "wb") as f:
			image.save(f, "JPEG")

		print("Success")
	except Exception as e:
		print('FAILED -', e)

urls = get_images_from_google(wd, 1, 15)

for i, url in enumerate(urls):
	download_image("imgs/", url, str(i) + ".jpg")

wd.quit()