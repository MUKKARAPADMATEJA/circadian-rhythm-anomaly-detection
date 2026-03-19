import xml.etree.ElementTree as ET
import csv
import sys
import os
from datetime import datetime

def parse_health_xml(xml_path, output_csv):
    print(f"Parsing {xml_path}...")
    
    # We only care about Heart Rate and Step Count for Circadian Rhythm analysis
    target_types = {
        'HKQuantityTypeIdentifierHeartRate': 'HeartRate',
        'HKQuantityTypeIdentifierStepCount': 'StepCount'
    }
    
    # Open CSV for writing
    with open(output_csv, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(['type', 'value', 'start_date', 'end_date'])
        
        count = 0
        extracted_count = 0
        
        try:
            # Iteratively parse XML
            context = ET.iterparse(xml_path, events=('start', 'end'))
            for event, elem in context:
                if event == 'end' and elem.tag == 'Record':
                    record_type = elem.attrib.get('type')
                    
                    if record_type in target_types:
                        val = elem.attrib.get('value')
                        start_date = elem.attrib.get('startDate')
                        end_date = elem.attrib.get('endDate')
                        
                        if val and start_date and end_date:
                            writer.writerow([target_types[record_type], val, start_date, end_date])
                            extracted_count += 1
                            
                    # Clear element from memory
                    elem.clear()
                
                # Check status periodically
                count += 1
                if count % 1000000 == 0:
                    print(f"Processed {count} elements, extracted {extracted_count} target records...")
            
            print(f"Finished parsing! Extracted a total of {extracted_count} records.")
            return True
            
        except Exception as e:
            print(f"Error reading XML: {e}")
            return False

if __name__ == "__main__":
    xml_path = r"C:\Users\MUKKARA PADMA TEJA\Downloads\export\apple_health_export\export.xml"
    output_path = r"C:\Users\MUKKARA PADMA TEJA\.gemini\antigravity\scratch\health_data_parsed.csv"
    
    if os.path.exists(xml_path):
        parse_health_xml(xml_path, output_path)
    else:
        print(f"Could not find XML at: {xml_path}")
