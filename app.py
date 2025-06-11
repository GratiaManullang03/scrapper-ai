import os
import json
import re
import time
from flask import Flask, render_template, request, jsonify
from playwright.sync_api import sync_playwright, Error
import together
from dotenv import load_dotenv
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple

# Load environment variables
load_dotenv()

# Configuration
app = Flask(__name__)
BROWSER_DEBUG_PORT = "http://localhost:9222"

# --- Instruction Parser ---
class InstructionParser:
    """Parse user instructions to understand exactly what they want"""
    
    @staticmethod
    def parse_table_request(instruction: str) -> Dict[str, Any]:
        """Parse instruction for table data extraction"""
        result = {
            'table_identifier': None,
            'requested_columns': [],
            'is_table_request': False,
            'extract_all': False
        }
        
        instruction_lower = instruction.lower()
        
        # Check if it's a table request
        table_keywords = ['tabel', 'table', 'dari tabel', 'from table', 'dalam tabel', 'in table']
        if any(keyword in instruction_lower for keyword in table_keywords):
            result['is_table_request'] = True
        
        # Extract table name/identifier
        table_patterns = [
            r'(?:dari tabel|from table|dalam tabel|in table)\s+["\']?([^,"\'.]+)["\']?',
            r'tabel\s+["\']?([^,"\'.]+)["\']?',
            r'table\s+["\']?([^,"\'.]+)["\']?'
        ]
        
        for pattern in table_patterns:
            match = re.search(pattern, instruction_lower)
            if match:
                result['table_identifier'] = match.group(1).strip()
                break
        
        # Check if user wants all data
        if any(word in instruction_lower for word in ['semua', 'all', 'seluruh', 'every']):
            result['extract_all'] = True
        
        # Extract requested columns
        # Remove the table part to focus on columns
        column_part = instruction
        for pattern in table_patterns:
            column_part = re.sub(pattern, '', column_part, flags=re.IGNORECASE)
        
        # Common patterns for listing columns
        column_patterns = [
            r'(?:ambil|get|extract|ekstrak)\s+(.+?)(?:\.|$)',
            r'(?:kolom|columns?|fields?)\s*:?\s*(.+?)(?:\.|$)',
            r'(?:yaitu|namely|seperti|such as|including)\s*:?\s*(.+?)(?:\.|$)'
        ]
        
        columns_text = ""
        for pattern in column_patterns:
            match = re.search(pattern, column_part, re.IGNORECASE)
            if match:
                columns_text = match.group(1)
                break
        
        if not columns_text and result['extract_all']:
            # If no specific columns but "all" is mentioned
            columns_text = column_part
        
        # Parse column names
        if columns_text:
            # Split by common delimiters
            delimiters = [',', ' dan ', ' and ', ', dan ', ', and ', ';']
            columns = [columns_text]
            
            for delimiter in delimiters:
                new_columns = []
                for col in columns:
                    new_columns.extend(col.split(delimiter))
                columns = new_columns
            
            # Clean up column names
            for col in columns:
                cleaned = col.strip().strip('"\'').strip()
                if cleaned and len(cleaned) > 2:  # Avoid single letters
                    result['requested_columns'].append(cleaned)
        
        return result
    
    @staticmethod
    def parse_single_values(instruction: str) -> List[str]:
        """Parse instruction for single value extraction"""
        values = []
        
        # Common patterns for single values
        patterns = [
            r'(?:cari|find|get|ambil|ekstrak|extract)\s+(.+?)(?:\.|$)',
            r'(?:nilai|value|jumlah|amount)\s+(.+?)(?:\.|$)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, instruction, re.IGNORECASE)
            if match:
                text = match.group(1)
                # Split by common delimiters
                parts = re.split(r',| dan | and |;', text)
                values.extend([p.strip() for p in parts if p.strip()])
        
        return values

# --- Smart Column Matcher ---
class ColumnMatcher:
    """Match requested columns with actual table headers"""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text for comparison"""
        # Remove special characters and convert to lowercase
        normalized = re.sub(r'[^\w\s]', '', text.lower())
        # Remove extra spaces
        normalized = ' '.join(normalized.split())
        return normalized
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """Calculate similarity between two texts"""
        norm1 = ColumnMatcher.normalize_text(text1)
        norm2 = ColumnMatcher.normalize_text(text2)
        
        # Exact match
        if norm1 == norm2:
            return 1.0
        
        # Contains match
        if norm1 in norm2 or norm2 in norm1:
            return 0.8
        
        # Word overlap
        words1 = set(norm1.split())
        words2 = set(norm2.split())
        if words1 and words2:
            overlap = len(words1.intersection(words2))
            total = len(words1.union(words2))
            return overlap / total if total > 0 else 0
        
        return 0
    
    @staticmethod
    def match_columns(requested: List[str], available: List[str]) -> Dict[str, str]:
        """Match requested columns to available headers"""
        matches = {}
        
        for req_col in requested:
            best_match = None
            best_score = 0
            
            for avail_col in available:
                score = ColumnMatcher.calculate_similarity(req_col, avail_col)
                if score > best_score and score > 0.3:  # Minimum threshold
                    best_score = score
                    best_match = avail_col
            
            if best_match:
                matches[req_col] = best_match
        
        return matches

# --- Enhanced Table Extractor ---
class FocusedTableExtractor:
    """Extract only requested data from tables"""
    
    @staticmethod
    def find_table_by_context(frame_locator, table_identifier: Optional[str]) -> Any:
        """Find table based on context clues"""
        
        # If table identifier provided, look for it
        if table_identifier:
            # Look for heading containing the identifier
            heading_patterns = [
                f"text=/{table_identifier}/i",
                f"h1:has-text('{table_identifier}')",
                f"h2:has-text('{table_identifier}')",
                f"h3:has-text('{table_identifier}')",
                f"*:has-text('{table_identifier}')"
            ]
            
            for pattern in heading_patterns:
                try:
                    headings = frame_locator.locator(pattern)
                    if headings.count() > 0:
                        # Look for table near this heading
                        heading = headings.first
                        # Try to find table after heading
                        nearby_table = heading.locator("xpath=following::table[1]")
                        if nearby_table.count() > 0:
                            return nearby_table.first
                        
                        # Try parent then table
                        parent = heading.locator("..")
                        table_in_parent = parent.locator("table")
                        if table_in_parent.count() > 0:
                            return table_in_parent.first
                except:
                    continue
        
        # Fallback: find any table
        tables = frame_locator.locator("table")
        if tables.count() > 0:
            # If multiple tables, try to find the most relevant one
            for i in range(tables.count()):
                table = tables.nth(i)
                # Check if table has reasonable content
                rows = table.locator("tr")
                if rows.count() > 1:  # At least header + 1 data row
                    return table
            
            # If no good table found, return first
            return tables.first
        
        return None
    
    @staticmethod
    def extract_table_headers(table_element) -> List[str]:
        """Extract headers from table"""
        headers = []
        
        # Try different header patterns
        header_patterns = [
            "thead th",
            "thead td", 
            "tr:first-child th",
            "tr:first-child td",
            "tr th"
        ]
        
        for pattern in header_patterns:
            try:
                header_elements = table_element.locator(pattern)
                if header_elements.count() > 0:
                    for i in range(header_elements.count()):
                        header_text = header_elements.nth(i).text_content().strip()
                        if header_text:
                            headers.append(header_text)
                    
                    if headers:
                        return headers
            except:
                continue
        
        return headers
    
    @staticmethod
    def extract_focused_data(frame_locator, instruction: str) -> List[Dict[str, Any]]:
        """Extract only the requested data from tables"""
        
        # Parse instruction
        parsed = InstructionParser.parse_table_request(instruction)
        
        if not parsed['is_table_request']:
            return []
        
        print(f"Parsed instruction: {json.dumps(parsed, indent=2, ensure_ascii=False)}")
        
        # Find the table
        table = FocusedTableExtractor.find_table_by_context(
            frame_locator, 
            parsed['table_identifier']
        )
        
        if not table:
            print("No table found")
            return []
        
        # Extract headers
        headers = FocusedTableExtractor.extract_table_headers(table)
        print(f"Found headers: {headers}")
        
        # Match requested columns to actual headers
        if parsed['requested_columns'] and headers:
            column_mapping = ColumnMatcher.match_columns(
                parsed['requested_columns'], 
                headers
            )
            print(f"Column mapping: {json.dumps(column_mapping, indent=2, ensure_ascii=False)}")
        else:
            column_mapping = {}
        
        # Extract data
        results = []
        
        # Find data rows
        row_patterns = [
            "tbody tr",
            "tr:has(td)",
            "tr:not(:first-child)"
        ]
        
        rows = None
        for pattern in row_patterns:
            try:
                rows = table.locator(pattern)
                if rows.count() > 0:
                    break
            except:
                continue
        
        if not rows:
            return []
        
        # Extract data from rows
        for i in range(min(rows.count(), 100)):  # Limit to 100 rows
            try:
                row = rows.nth(i)
                cells = row.locator("td")
                
                if cells.count() == 0:
                    continue
                
                row_data = {}
                
                # If we have column mapping, extract only requested columns
                if column_mapping:
                    for requested_name, actual_header in column_mapping.items():
                        # Find column index
                        try:
                            col_index = headers.index(actual_header)
                            if col_index < cells.count():
                                cell_text = cells.nth(col_index).text_content().strip()
                                row_data[requested_name] = cell_text
                        except:
                            continue
                
                # If no mapping or extract_all, get all columns
                elif parsed['extract_all'] or not parsed['requested_columns']:
                    for j in range(cells.count()):
                        if j < len(headers):
                            header = headers[j]
                            cell_text = cells.nth(j).text_content().strip()
                            row_data[header] = cell_text
                
                if row_data:
                    results.append(row_data)
                    
            except Exception as e:
                print(f"Error extracting row {i}: {e}")
                continue
        
        print(f"Extracted {len(results)} rows with focused data")
        return results

# --- Main Routes ---
@app.route('/')
def index():
    """Display main HTML page"""
    return render_template('index.html')

@app.route('/scrape', methods=['POST'])
def scrape():
    """Smart scraping endpoint with focused extraction"""
    
    print("\n=== FOCUSED SMART SCRAPER ===")
    
    data = request.form
    url = data.get('url')
    instruction = data.get('instruction')
    
    if not url or not instruction:
        return jsonify({"error": "URL and instruction are required"}), 400
    
    if not os.getenv("TOGETHER_AI_API_KEY"):
        return jsonify({"error": "TOGETHER_AI_API_KEY not found"}), 500
    
    print(f"URL: {url}")
    print(f"Instruction: {instruction}")
    
    try:
        with sync_playwright() as p:
            # Connect to browser
            print("Connecting to browser...")
            browser = p.chromium.connect_over_cdp(BROWSER_DEBUG_PORT)
            context = browser.contexts[0]
            page = context.new_page()
            
            # Navigate to URL
            print(f"Navigating to {url}...")
            page.goto(url, wait_until='domcontentloaded', timeout=60000)
            page.wait_for_timeout(3000)
            
            # Check for iframe
            frame_locator = page
            iframe_selectors = [".space-iframe", "iframe#main", "iframe[src]"]
            
            for selector in iframe_selectors:
                if page.locator(selector).count() > 0:
                    print(f"Found iframe: {selector}")
                    page.wait_for_selector(selector, timeout=30000)
                    frame_locator = page.frame_locator(selector)
                    frame_locator.locator("body").wait_for(timeout=30000)
                    page.wait_for_timeout(2000)
                    break
            
            # Try focused table extraction first
            print("\nAttempting focused table extraction...")
            results = FocusedTableExtractor.extract_focused_data(frame_locator, instruction)
            
            if results:
                print(f"\nExtraction successful: {len(results)} rows")
                page.close()
                return jsonify(results)
            
            # If focused extraction fails, fall back to AI method
            print("\nFocused extraction failed, using AI assistance...")
            
            # Get HTML content
            if frame_locator != page:
                html_content = frame_locator.locator("body").inner_html()
            else:
                html_content = page.content()
            
            # Use AI to generate selectors
            client = together.Together(api_key=os.getenv("TOGETHER_AI_API_KEY"))
            
            # Create focused prompt
            parsed_instruction = InstructionParser.parse_table_request(instruction)
            
            prompt = f"""
            Create CSS selectors to extract ONLY the specific data requested by the user.
            
            User Instruction: "{instruction}"
            
            Requested Columns: {json.dumps(parsed_instruction['requested_columns'])}
            
            Important:
            1. Create selectors ONLY for the requested columns
            2. Return a JSON object with keys matching the requested column names
            3. If the user asks for "ID Pesanan", use exactly "ID Pesanan" as the key
            4. Focus on the specific table mentioned in the instruction
            
            HTML (first 10000 chars):
            ```html
            {html_content[:10000]}
            ```
            
            Return ONLY a JSON object with selectors for the requested data.
            Example: {{"ID Pesanan": "selector", "Nama Pelanggan": "selector"}}
            """
            
            response = client.chat.completions.create(
                model="meta-llama/Llama-3-70b-chat-hf",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
                response_format={"type": "json_object"}
            )
            
            selector_data = json.loads(response.choices[0].message.content)
            print(f"AI selectors: {json.dumps(selector_data, indent=2, ensure_ascii=False)}")
            
            # Extract using AI selectors
            if any('tbody' in str(v) or 'tr' in str(v) for v in selector_data.values()):
                # Table extraction
                all_rows = []
                
                # Find how many rows exist
                first_selector = list(selector_data.values())[0]
                base_selector = first_selector.split('td')[0] + 'td'
                
                try:
                    test_elements = frame_locator.locator(base_selector)
                    # Estimate row count
                    total_cells = test_elements.count()
                    cols_per_row = len(selector_data)
                    estimated_rows = total_cells // cols_per_row if cols_per_row > 0 else 0
                    
                    print(f"Estimated {estimated_rows} rows")
                    
                    for row_idx in range(min(estimated_rows, 100)):
                        row_data = {}
                        
                        for key, selector in selector_data.items():
                            try:
                                # Modify selector for specific row
                                if 'tr:nth-child' not in selector:
                                    # Add row index
                                    if 'tbody tr' in selector:
                                        row_selector = selector.replace('tbody tr', f'tbody tr:nth-child({row_idx + 1})')
                                    else:
                                        row_selector = f"tbody tr:nth-child({row_idx + 1}) {selector.split(')')[-1]}"
                                else:
                                    # Update existing row index
                                    row_selector = re.sub(r'tr:nth-child\(\d+\)', f'tr:nth-child({row_idx + 1})', selector)
                                
                                elem = frame_locator.locator(row_selector)
                                if elem.count() > 0:
                                    text = elem.first.text_content()
                                    row_data[key] = text.strip() if text else ""
                            except:
                                continue
                        
                        if any(v for v in row_data.values()):
                            all_rows.append(row_data)
                    
                    if all_rows:
                        print(f"Extracted {len(all_rows)} rows")
                        page.close()
                        return jsonify(all_rows)
                        
                except Exception as e:
                    print(f"Table extraction error: {e}")
            
            # Single value extraction
            extracted_data = {}
            for key, selector in selector_data.items():
                try:
                    elem = frame_locator.locator(selector)
                    if elem.count() > 0:
                        text = elem.first.text_content()
                        extracted_data[key] = text.strip() if text else ""
                    else:
                        extracted_data[key] = "Not found"
                except:
                    extracted_data[key] = "Error"
            
            page.close()
            return jsonify([extracted_data])
            
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, port=5001)