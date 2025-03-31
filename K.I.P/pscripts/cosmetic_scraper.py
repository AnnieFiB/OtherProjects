import requests
from bs4 import BeautifulSoup
import pandas as pd
from urllib.parse import urlparse
import time
import random
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import logging
import pytesseract
from PIL import Image
import io
import os
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('scraper.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class CaptchaSolver:
    def __init__(self):
        self.use_ocr = True
        self.captcha_timeout = 120
        self.captcha_images_dir = 'captcha_images'
        os.makedirs(self.captcha_images_dir, exist_ok=True)

    def solve_captcha(self, image_element, url):
        """Solve CAPTCHA using OCR or manual input"""
        try:
            timestamp = int(time.time())
            captcha_path = os.path.join(self.captcha_images_dir, f'captcha_{timestamp}.png')
            
            image_data = image_element.screenshot_as_png
            with open(captcha_path, 'wb') as f:
                f.write(image_data)
            
            if self.use_ocr:
                captcha_text = self._solve_with_ocr(captcha_path)
                if captcha_text:
                    return captcha_text
            
            return self._solve_manually(captcha_path, url)
            
        except Exception as e:
            logger.error(f"CAPTCHA solving failed: {str(e)}")
            return None

    def _solve_with_ocr(self, image_path):
        """Solve CAPTCHA using Tesseract OCR"""
        try:
            image = Image.open(image_path)
            image = image.convert('L')
            image = image.point(lambda x: 0 if x < 128 else 255, '1')
            captcha_text = pytesseract.image_to_string(image).strip()
            if captcha_text:
                logger.info(f"OCR solved CAPTCHA: {captcha_text}")
                return captcha_text
            return None
        except Exception as e:
            logger.warning(f"OCR failed: {str(e)}")
            return None

    def _solve_manually(self, image_path, url):
        """Prompt user to solve CAPTCHA manually"""
        try:
            if sys.platform == 'win32':
                os.startfile(image_path)
            elif sys.platform == 'darwin':
                os.system(f'open "{image_path}"')
            else:
                os.system(f'xdg-open "{image_path}"')
            
            logger.warning(f"Manual CAPTCHA solving required for {url}")
            logger.warning(f"CAPTCHA image saved to: {image_path}")
            
            start_time = time.time()
            while time.time() - start_time < self.captcha_timeout:
                captcha_text = input("Enter CAPTCHA text (or 'skip' to skip): ").strip()
                if captcha_text.lower() == 'skip':
                    return None
                if captcha_text:
                    return captcha_text
                print("Invalid input, try again")
            
            logger.warning("CAPTCHA input timeout reached")
            return None
        except Exception as e:
            logger.error(f"Manual CAPTCHA solving failed: {str(e)}")
            return None

class CosmeticScraper:
    def __init__(self):
        self.session = requests.Session()
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        self.selenium_driver = None
        self.captcha_solver = CaptchaSolver()
        self.initialize_selenium()

    def initialize_selenium(self):
        """Initialize Selenium WebDriver"""
        try:
            chrome_options = Options()
            # chrome_options.add_argument("--headless")  # Disable for debugging
            chrome_options.add_argument("--disable-gpu")
            chrome_options.add_argument("--no-sandbox")
            chrome_options.add_argument("--disable-dev-shm-usage")
            chrome_options.add_experimental_option("excludeSwitches", ["enable-automation"])
            chrome_options.add_experimental_option('useAutomationExtension', False)
            
            self.selenium_driver = webdriver.Chrome(options=chrome_options)
            self.selenium_driver.set_page_load_timeout(45)
        except Exception as e:
            logger.error(f"Failed to initialize Selenium: {str(e)}")
            self.selenium_driver = None

    def close_selenium(self):
        if self.selenium_driver:
            self.selenium_driver.quit()

    def handle_captcha(self, driver, url):
        """Detect and handle CAPTCHA challenges"""
        try:
            captcha_selectors = [
                ('iframe[src*="captcha"]', 'iframe'),
                ('div.g-recaptcha', 'div'),
                ('img.captcha-img', 'img'),
                ('#captcha', 'div'),
                ('img[src*="captcha"]', 'img'),
                ('div.recaptcha', 'div')
            ]
            
            for selector, element_type in captcha_selectors:
                captcha_elements = driver.find_elements(By.CSS_SELECTOR, selector)
                if captcha_elements:
                    logger.warning(f"CAPTCHA detected on {url}, attempting to solve...")
                    
                    if 'iframe' in element_type:
                        driver.switch_to.frame(captcha_elements[0])
                        captcha_image = driver.find_elements(By.CSS_SELECTOR, 'img')
                        if captcha_image:
                            solution = self.captcha_solver.solve_captcha(captcha_image[0], url)
                            if solution:
                                input_field = driver.find_elements(By.CSS_SELECTOR, 'input[type="text"]')
                                if input_field:
                                    input_field[0].clear()
                                    input_field[0].send_keys(solution)
                                    submit_button = driver.find_elements(By.CSS_SELECTOR, 'button[type="submit"]')
                                    if submit_button:
                                        submit_button[0].click()
                                        time.sleep(3)
                                        driver.switch_to.default_content()
                                        return True
                        driver.switch_to.default_content()
                    
                    elif 'img' in element_type:
                        solution = self.captcha_solver.solve_captcha(captcha_elements[0], url)
                        if solution:
                            input_xpaths = [
                                '//input[contains(@name, "captcha")]',
                                '//input[contains(@id, "captcha")]',
                                '//input[@type="text"]'
                            ]
                            for xpath in input_xpaths:
                                input_field = driver.find_elements(By.XPATH, xpath)
                                if input_field:
                                    input_field[0].clear()
                                    input_field[0].send_keys(solution)
                                    submit_buttons = [
                                        '//button[contains(@name, "submit")]',
                                        '//button[contains(@id, "submit")]',
                                        '//input[@type="submit"]'
                                    ]
                                    for btn_xpath in submit_buttons:
                                        submit_button = driver.find_elements(By.XPATH, btn_xpath)
                                        if submit_button:
                                            submit_button[0].click()
                                            time.sleep(3)
                                            return True
                                    break
                    
                    elif 'div' in element_type:
                        logger.warning("reCAPTCHA detected - attempting manual solution")
                        captcha_image = driver.find_elements(By.CSS_SELECTOR, 'img')
                        if captcha_image:
                            solution = self.captcha_solver.solve_captcha(captcha_image[0], url)
                            if solution:
                                input_field = driver.find_elements(By.CSS_SELECTOR, 'textarea#g-recaptcha-response')
                                if input_field:
                                    driver.execute_script("arguments[0].style.display = 'block';", input_field[0])
                                    input_field[0].clear()
                                    input_field[0].send_keys(solution)
                                    time.sleep(1)
                                    submit_button = driver.find_elements(By.CSS_SELECTOR, 'button[type="submit"]')
                                    if submit_button:
                                        submit_button[0].click()
                                        time.sleep(3)
                                        return True
            
            return False
        except Exception as e:
            logger.error(f"CAPTCHA handling failed: {str(e)}")
            return False

    def get_page_content(self, url):
        """Get page content with CAPTCHA handling"""
        domain = urlparse(url).netloc
        
        js_sites = [
            'sephora.com.mx', 'maccosmetics.com.mx', 'ulta.com',
            'mercadolibre.com.mx', 'liverpool.com.mx', 'sanborns.com.mx',
            'elpalaciodehierro.com', 'walmart.com.mx'
        ]
        
        if any(js_site in domain for js_site in js_sites) and self.selenium_driver:
            try:
                logger.info(f"Using Selenium for {url}")
                self.selenium_driver.get(url)
                
                if self.handle_captcha(self.selenium_driver, url):
                    logger.info("CAPTCHA solved, retrying page load")
                    self.selenium_driver.get(url)
                
                WebDriverWait(self.selenium_driver, 15).until(
                    EC.presence_of_element_located((By.TAG_NAME, "body"))
                )
                return BeautifulSoup(self.selenium_driver.page_source, 'html.parser')
            except Exception as e:
                logger.error(f"Selenium failed for {url}: {str(e)}")
                return None
        
        try:
            logger.info(f"Using requests for {url}")
            response = self.session.get(url, headers=self.headers, timeout=15)
            
            if any(keyword in response.text.lower() for keyword in ['captcha', 'recaptcha', 'verification']):
                logger.warning(f"CAPTCHA detected on {url} - requires Selenium")
                return None
                
            response.raise_for_status()
            return BeautifulSoup(response.content, 'html.parser')
        except Exception as e:
            logger.error(f"Requests failed for {url}: {str(e)}")
            return None

    def scrape_generic_product(self, soup, url):
        """Generic scraping logic"""
        product_data = {
            'website': url,
            'product_name': None,
            'brand': None,
            'description': None,
            'price': None,
            'original_price': None,
            'quantity': None,
            'available': None,
            'image_url': None,
            'scrape_timestamp': pd.Timestamp.now().isoformat()
        }

        product_data['product_name'] = self.extract_field(soup, [
            'h1.product-title', 'h1.product-name', 'h1.title', 
            'h1[itemprop="name"]', 'h1.product__title', 'h1.product-title',
            'h1.productName', 'h1.product_name'
        ])

        product_data['brand'] = self.extract_field(soup, [
            'a.brand', 'span.brand', 'div.brand', 
            'meta[itemprop="brand"]', 'span[itemprop="brand"]',
            'div.product-brand', 'a[href*="brand"]'
        ], attribute='content')

        product_data['description'] = self.extract_field(soup, [
            'div.product-description', 'div.description', 
            'div[itemprop="description"]', 'div.product__description',
            'div.details', 'div.product-info', 'div.product-details'
        ])

        product_data['price'] = self.extract_field(soup, [
            'span.price', 'span.product-price', 'meta[itemprop="price"]',
            'span[itemprop="price"]', 'span.money', 'div.price',
            'span.final-price', 'span.sale-price'
        ], attribute='content')

        product_data['original_price'] = self.extract_field(soup, [
            'span.compare-at-price', 'span.original-price',
            'span.was-price', 'del.price', 'span.old-price',
            'span.regular-price'
        ])

        product_data['available'] = self.check_availability(soup)

        product_data['image_url'] = self.extract_field(soup, [
            'img.product-image', 'img[itemprop="image"]',
            'div.product-image img', 'meta[property="og:image"]',
            'img.main-image', 'img.primary-image'
        ], attribute='src') or self.extract_field(soup, [
            'meta[property="og:image"]'
        ], attribute='content')

        return product_data

    def extract_field(self, soup, selectors, attribute=None):
        """Extract field using multiple possible selectors"""
        for selector in selectors:
            try:
                element = soup.select_one(selector)
                if element:
                    if attribute:
                        return element.get(attribute, '').strip()
                    return element.get_text(' ', strip=True)
            except Exception as e:
                logger.debug(f"Selector {selector} failed: {str(e)}")
                continue
        return None

    def check_availability(self, soup):
        """Check product availability"""
        available_text = ['agotado', 'sold out', 'no disponible', 'out of stock', 'no hay existencias']
        unavailable_elements = soup.find_all(string=available_text)
        return len(unavailable_elements) == 0

    def scrape_specific_website(self, url, soup):
        """Special handling for known websites"""
        domain = urlparse(url).netloc.lower()
        
        if 'sephora.com.mx' in domain:
            return self.scrape_sephora(soup, url)
        elif 'maccosmetics.com.mx' in domain:
            return self.scrape_mac(soup, url)
        elif 'mercadolibre.com.mx' in domain:
            return self.scrape_mercado_libre(soup, url)
        elif 'liverpool.com.mx' in domain:
            return self.scrape_liverpool(soup, url)
        
        return None

    def scrape_sephora(self, soup, url):
        try:
            data = {
                'product_name': self.extract_field(soup, ['h1.css-1wd7e2u', 'h1.product-name']),
                'brand': self.extract_field(soup, ['a.css-1d0x7lo', 'a.brand-link']),
                'description': self.extract_field(soup, ['div.css-pz80c5', 'div.product-description']),
                'price': self.extract_field(soup, ['span.css-0', 'span.price']),
                'available': not bool(soup.select_one('button[aria-label="Agotado"]'))
            }
            return {**self.scrape_generic_product(soup, url), **data}
        except Exception as e:
            logger.error(f"Error scraping Sephora: {str(e)}")
            return None

    def scrape_mac(self, soup, url):
        try:
            data = {
                'product_name': self.extract_field(soup, ['h1.product-name']),
                'brand': 'MAC Cosmetics',
                'description': self.extract_field(soup, ['div.product-detail-description']),
                'price': self.extract_field(soup, ['span.price-sales']),
                'available': not bool(soup.select_one('div.out-of-stock'))
            }
            return {**self.scrape_generic_product(soup, url), **data}
        except Exception as e:
            logger.error(f"Error scraping MAC: {str(e)}")
            return None

    def scrape_mercado_libre(self, soup, url):
        try:
            data = {
                'product_name': self.extract_field(soup, ['h1.ui-pdp-title']),
                'brand': self.extract_field(soup, ['th:contains("Marca") + td']),
                'description': self.extract_field(soup, ['p.ui-pdp-description__content']),
                'price': self.extract_field(soup, ['span.andes-money-amount__fraction']),
                'available': not bool(soup.select_one('p.ui-pdp-stock-information__title'))
            }
            return {**self.scrape_generic_product(soup, url), **data}
        except Exception as e:
            logger.error(f"Error scraping Mercado Libre: {str(e)}")
            return None

    def scrape_liverpool(self, soup, url):
        try:
            data = {
                'product_name': self.extract_field(soup, ['h1.product-name']),
                'brand': self.extract_field(soup, ['a.brand-name']),
                'description': self.extract_field(soup, ['div.product-description']),
                'price': self.extract_field(soup, ['span.price-sales']),
                'available': not bool(soup.select_one('div.out-of-stock'))
            }
            return {**self.scrape_generic_product(soup, url), **data}
        except Exception as e:
            logger.error(f"Error scraping Liverpool: {str(e)}")
            return None

    def scrape_website(self, url):
        """Main scraping function"""
        soup = self.get_page_content(url)
        if not soup:
            return None
        
        product_data = self.scrape_specific_website(url, soup)
        if product_data:
            return product_data
        
        return self.scrape_generic_product(soup, url)

def main():
    try:
        df = pd.read_csv('websites.csv')
        websites = df['Website'].tolist()
    except Exception as e:
        logger.error(f"Failed to read input file: {str(e)}")
        return

    scraper = CosmeticScraper()
    results = []
    failed_urls = []
    
    for idx, url in enumerate(websites):
        logger.info(f"Processing {idx+1}/{len(websites)}: {url}")
        time.sleep(random.uniform(2, 5))
        
        try:
            product_data = scraper.scrape_website(url)
            if product_data:
                results.append(product_data)
                logger.info(f"Scraped: {product_data.get('product_name')}")
            else:
                failed_urls.append(url)
        except Exception as e:
            logger.error(f"Error on {url}: {str(e)}")
            failed_urls.append(url)

        if idx > 0 and idx % 5 == 0:
            pd.DataFrame(results).to_csv('cosmetic_products_results.csv', index=False)
            if failed_urls:
                pd.DataFrame({'failed_urls': failed_urls}).to_csv('failed_urls.csv', index=False)

    pd.DataFrame(results).to_csv('cosmetic_products_results.csv', index=False)
    if failed_urls:
        pd.DataFrame({'failed_urls': failed_urls}).to_csv('failed_urls.csv', index=False)

    scraper.close_selenium()
    logger.info(f"Scraping completed. Results: {len(results)} successful, {len(failed_urls)} failed.")

if __name__ == "__main__":
    try:
        pytesseract.get_tesseract_version()
    except:
        logger.warning("Tesseract OCR not installed. CAPTCHA solving will be manual only.")
        logger.warning("Install Tesseract for automatic CAPTCHA solving:")
        logger.warning("Windows: https://github.com/UB-Mannheim/tesseract/wiki")
        logger.warning("Mac: brew install tesseract")
        logger.warning("Linux: sudo apt install tesseract-ocr")
    
    try:
        from selenium import webdriver
        import pytesseract
        from PIL import Image
    except ImportError as e:
        logger.error(f"Missing required package: {str(e)}")
        logger.info("Install with: pip install selenium pytesseract pillow")
        sys.exit(1)
    
    main()