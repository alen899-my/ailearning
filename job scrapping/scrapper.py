import csv
import time
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.support.ui import WebDriverWait # 👈 Added for Explicit Wait
from selenium.webdriver.support import expected_conditions as EC # 👈 Added for Explicit Wait

options = Options()
# options.add_argument("--headless")   # uncomment if you want to see browser
driver = webdriver.Chrome(options=options)

# Define the CSV filename consistently
CSV_FILENAME = "test.csv"

def find_visible_mailto(timeout=3):
    """Poll for a visible mailto: anchor and return the email (string) or ''."""
    end = time.time() + timeout
    while time.time() < end:
        anchors = driver.find_elements(By.CSS_SELECTOR, "a[href^='mailto:']")
        for a in anchors:
            try:
                if a.is_displayed():
                    href = a.get_attribute("href") or ""
                    if href.startswith("mailto:"):
                        # Extract email, removing 'mailto:' prefix and any query parameters
                        return href.split("mailto:")[1].split("?")[0].strip()
                    text = a.text.strip()
                    if "@" in text:
                        return text
            except:
                continue
        time.sleep(0.25)
    return ""

def try_close_modal():
    """Try a few common selectors to close a modal; press ESC as fallback."""
    close_selectors = [
        "button.close", "button.btn-close", ".modal .close", ".modal .btn-close",
        "button[aria-label='Close']", ".close-modal", ".fa-times", "a[aria-label='Close']"
    ]
    for sel in close_selectors:
        els = driver.find_elements(By.CSS_SELECTOR, sel)
        for e in els:
            try:
                if e.is_displayed():
                    e.click()
                    time.sleep(0.25)
                    return True
            except:
                pass
    # fallback: send ESC
    try:
        body = driver.find_element(By.TAG_NAME, "body")
        body.send_keys(Keys.ESCAPE)
        time.sleep(0.2)
        return True
    except:
        return False

# Open the CSV file and start the scraping process
with open(CSV_FILENAME, "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Company Name", "Job Title", "HR Email"])

    # Loop through job listing pages
    for page in range(1, 22):   # change upper bound if needed
        listing_url = f"https://infopark.in/companies/job-search?page={page}"
        print(f"\n--- Scraping Page {page} ---")
        driver.get(listing_url)
        time.sleep(2.0)  # Extended wait for full page load and JS execution

        # Build list of entries (company, job) from the current page's table
        entries = []
        rows = driver.find_elements(By.CSS_SELECTOR, "tr")
        for row in rows:
            try:
                comp_el = row.find_element(By.CSS_SELECTOR, "td.date")
                head_cells = row.find_elements(By.CSS_SELECTOR, "td.head")
                if len(head_cells) >= 2:
                    company = comp_el.text.strip()
                    job = head_cells[1].text.strip()
                    entries.append((company, job))
            except:
                continue

        # Iterate entries and click the corresponding details button by index
        for idx, (company, job) in enumerate(entries):
            
            # CRITICAL: re-find buttons inside the loop to avoid StaleElementReferenceError
            buttons = driver.find_elements(By.CSS_SELECTOR, "button.btn-white-txt")
            if idx >= len(buttons):
                # no matching details button found; save row with empty email
                writer.writerow([company, job, ""])
                print(f"[{page}] {company} | {job} | (no details button)")
                continue

            btn = buttons[idx]
            
            try:
                # 1. Scroll the element into view (centered)
                driver.execute_script("arguments[0].scrollIntoView({block:'center'});", btn)
                
                # 2. Use Explicit Wait to ensure the button is ready to be clicked
                WebDriverWait(driver, 5).until(
                    EC.element_to_be_clickable(btn)
                )
                
                # store current url to detect navigation
                before = driver.current_url
                btn.click() # Perform the click
            
            except Exception as e:
                # This catches the 'click failed' scenario (including timeout from WebDriverWait)
                writer.writerow([company, job, ""])
                print(f"[{page}] {company} | {job} | (click failed)")
                # print(f"DEBUG Error: {e}") # Uncomment for detailed error logging
                continue

            time.sleep(0.8)   # Increased wait for modal/navigation/JS to load content

            # Check if click navigated to a new detail page
            if driver.current_url != before:
                hr_email = find_visible_mailto(timeout=3)
                writer.writerow([company, job, hr_email])
                print(f"[{page}] {company} | {job} | {hr_email} (navigated)")
                
                # go back to listing page
                try:
                    driver.back()
                    time.sleep(1.0) # Longer wait after going back
                except:
                    pass
            else:
                # Likely opened inline expansion or modal
                hr_email = find_visible_mailto(timeout=3)
                writer.writerow([company, job, hr_email])
                print(f"[{page}] {company} | {job} | {hr_email} (modal/inline)")
                
                # try to close modal/overlay (if any)
                try_close_modal()
                time.sleep(0.5) # Increased wait after closing modal

driver.quit()
print(f"\nDone. Data extracted and saved to {CSV_FILENAME}")