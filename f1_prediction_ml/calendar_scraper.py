import requests
from bs4 import BeautifulSoup
from colors import CYAN, YELLOW, MAGENTA, RESET

REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.5',
}

def get_f1_calendar(year):
    url = f'https://www.formula1.com/en/racing/{year}/'
    try:
        print(f'{CYAN}Fetching F1 calendar for {year} from {url}...{RESET}')
        response = requests.get(url, headers=REQUEST_HEADERS)

    except requests.exceptions.RequestException as e:
        print(f'{MAGENTA}Error fetching calendar for {year}: {e}{RESET}')
        return None
    
    soup = BeautifulSoup(response.content, 'html.parser')
    
    # Extract calendar data
    calendar_data = []
    events = soup.find_all('a', class_='group')
    
    print(f'{CYAN}Found {len(events)} events for {year}.{RESET}')
    
    for event in events:
        testing = event.find('span', class_='typography-module_technical-m-bold__JDsxP')
        future_events = event.find('span', class_='typography-module_technical-m-bold__JDsxP typography-module_lg_technical-l-bold__d8tzL')
        if future_events:
            event_date = testing.get_text(strip=True) 
            event_names = event.find('p', class_='typography-module_display-xl-bold__Gyl5W group-hover:underline')
            event_name = ''.join(
                text for text in event_names.strings
                if text.parent.name not in ('svg', 'title', 'path', 'g', 'defs', 'clipPath', 'rect', 'td')
            ).strip()
            calendar_data.append({'event': f'{event_name} - Upcoming', 'date': event_date})
        elif testing:
            event_date = event.find('span', class_='typography-module_technical-m-bold__JDsxP')
            event_date = event_date.get_text(strip=True) if event_date else 'Unknown Date'
            event_names = event.find('p', class_='typography-module_display-xl-bold__Gyl5W group-hover:underline')
            event_name = ''.join(
                text for text in event_names.strings
                if text.parent.name not in ('svg', 'title', 'path', 'g', 'defs', 'clipPath', 'rect', 'td')
            ).strip()
            calendar_data.append({'event': f'{event_name} - Testing', 'date': event_date})
        else:
            event_date = event.find('span', class_='typography-module_technical-xs-regular__-W0Gs')
            event_date = event_date.get_text(strip=True) if event_date else 'Unknown Date'
            event_names = event.find('p', class_='typography-module_display-xl-bold__Gyl5W group-hover:underline')
            event_name = ''.join(
                text for text in event_names.strings
                if text.parent.name not in ('svg', 'title', 'path', 'g', 'defs', 'clipPath', 'rect', 'td')
            ).strip()

            calendar_data.append({'event': event_name, 'date': event_date})
    
    print(f'{CYAN}Successfully fetched calendar for {year} with {len(calendar_data)} events.{RESET}')
    print(calendar_data)
    return calendar_data

if __name__ == "__main__":
    year = 2026
    calendar = get_f1_calendar(year)
    
    if calendar:
        print(f'{CYAN}F1 Calendar for {year}:{RESET}')
        for event in calendar:
            print(f"{YELLOW}{event['date']}: {event['event']}{RESET}")