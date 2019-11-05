from io import BytesIO
import requests as req
from PIL import Image

url = 'https://cdn.theatlantic.com/assets/media/img/mt/2019/02/shutterstock_559266193/lead_720_405.jpg?mod=1550855715'

def url_image(url):
    response = req.get(url)
    image = Image.open(BytesIO(response.content))
    return image


def main():

    image = url_image(url)
    # image_resize = image.resize((200,300))
    # image_transpose = image.transpose(True)
    image_rotate = image.rotate(20)
    image_rotate.show()

    
main()