import subprocess
import os
import re
import xml.etree.ElementTree as ET
import tempfile
from typing import Dict, List, Any

# Path to checkstyle jar file
CHECKSTYLE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "checkstyle.jar")
# Path to checkstyle configuration
CHECKSTYLE_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "checkstyle.xml")

def run_checkstyle(java_file_path: str) -> Dict[str, Any]:
    """
    Run Checkstyle on a Java file and return the results.
    
    Args:
        java_file_path: Path to the Java file to check
        
    Returns:
        A dictionary with checkstyle results:
        {
            "issues": [
                {
                    "line": line_number,
                    "message": error_message
                },
                ...
            ]
        }
    """
    result = {
        "issues": []
    }
    
    try:
        # Ensure the file exists
        if not os.path.exists(java_file_path):
            result["issues"].append({
                "line": 0,
                "message": f"File not found: {java_file_path}"
            })
            return result
        
        # Ensure checkstyle is available
        if not os.path.exists(CHECKSTYLE_PATH):
            if not download_checkstyle():
                result["issues"].append({
                    "line": 0,
                    "message": "Checkstyle is not available. Please install it manually."
                })
                return result
        
        # Create a temporary file for XML output
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as temp_xml:
            temp_xml_path = temp_xml.name
        
        # Run checkstyle with XML output format
        process = subprocess.run(
            [
                "java", "-jar", CHECKSTYLE_PATH,
                "-c", CHECKSTYLE_CONFIG,
                "-f", "xml",
                "-o", temp_xml_path,
                java_file_path
            ],
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Parse the XML output
        if os.path.exists(temp_xml_path) and os.path.getsize(temp_xml_path) > 0:
            try:
                tree = ET.parse(temp_xml_path)
                root = tree.getroot()
                
                for file_elem in root.findall(".//file"):
                    for error_elem in file_elem.findall(".//error"):
                        line = int(error_elem.get("line", 0))
                        message = error_elem.get("message", "Unknown error")
                        
                        result["issues"].append({
                            "line": line,
                            "message": message
                        })
            except ET.ParseError:
                # If XML parsing fails, try to extract information from command output
                if process.stderr:
                    for line in process.stderr.split("\n"):
                        match = re.search(r"\[(\d+)\]\s*(.*)", line)
                        if match:
                            line_num = int(match.group(1))
                            message = match.group(2)
                            result["issues"].append({
                                "line": line_num,
                                "message": message
                            })
        
        # Clean up temporary file
        if os.path.exists(temp_xml_path):
            os.unlink(temp_xml_path)
        
        return result
    
    except subprocess.TimeoutExpired:
        result["issues"].append({
            "line": 0,
            "message": "Checkstyle analysis timed out after 30 seconds"
        })
        return result
    
    except FileNotFoundError:
        result["issues"].append({
            "line": 0,
            "message": "Java not found. Please make sure Java is installed."
        })
        return result
    
    except Exception as e:
        result["issues"].append({
            "line": 0,
            "message": f"An error occurred during Checkstyle analysis: {str(e)}"
        })
        return result

def download_checkstyle() -> bool:
    """
    Download the Checkstyle JAR file.
    
    Returns:
        True if download was successful, False otherwise
    """
    try:
        # Create data directory if it doesn't exist
        data_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")
        os.makedirs(data_dir, exist_ok=True)
        
        # Download the Checkstyle JAR
        checkstyle_url = "https://github.com/checkstyle/checkstyle/releases/download/checkstyle-10.3.3/checkstyle-10.3.3-all.jar"
        
        process = subprocess.run(
            ["curl", "-L", checkstyle_url, "-o", CHECKSTYLE_PATH],
            capture_output=True,
            text=True
        )
        
        if process.returncode != 0:
            print(f"Failed to download Checkstyle: {process.stderr}")
            return False
        
        # Create a basic Checkstyle configuration file
        create_default_checkstyle_config()
        
        return os.path.exists(CHECKSTYLE_PATH)
    
    except Exception as e:
        print(f"An error occurred while downloading Checkstyle: {str(e)}")
        return False

def create_default_checkstyle_config() -> None:
    """
    Create a default Checkstyle configuration file.
    """
    config_content = """<?xml version="1.0"?>
<!DOCTYPE module PUBLIC
          "-//Checkstyle//DTD Checkstyle Configuration 1.3//EN"
          "https://checkstyle.org/dtds/configuration_1_3.dtd">

<module name="Checker">
    <property name="severity" value="warning"/>

    <module name="TreeWalker">
        <!-- Naming Conventions -->
        <module name="ConstantName"/>
        <module name="LocalFinalVariableName"/>
        <module name="LocalVariableName"/>
        <module name="MemberName"/>
        <module name="MethodName"/>
        <module name="PackageName"/>
        <module name="ParameterName"/>
        <module name="StaticVariableName"/>
        <module name="TypeName"/>

        <!-- Imports -->
        <module name="AvoidStarImport"/>
        <module name="IllegalImport"/>
        <module name="RedundantImport"/>
        <module name="UnusedImports"/>

        <!-- Size Violations -->
        <module name="MethodLength"/>
        <module name="ParameterNumber"/>

        <!-- Whitespace -->
        <module name="EmptyForIteratorPad"/>
        <module name="GenericWhitespace"/>
        <module name="MethodParamPad"/>
        <module name="NoWhitespaceAfter"/>
        <module name="NoWhitespaceBefore"/>
        <module name="ParenPad"/>
        <module name="TypecastParenPad"/>
        <module name="WhitespaceAfter"/>
        <module name="WhitespaceAround"/>

        <!-- Coding -->
        <module name="EmptyStatement"/>
        <module name="EqualsHashCode"/>
        <module name="HiddenField"/>
        <module name="IllegalInstantiation"/>
        <module name="InnerAssignment"/>
        <module name="MagicNumber"/>
        <module name="MissingSwitchDefault"/>
        <module name="MultipleVariableDeclarations"/>
        <module name="SimplifyBooleanExpression"/>
        <module name="SimplifyBooleanReturn"/>

        <!-- Design -->
        <module name="FinalClass"/>
        <module name="HideUtilityClassConstructor"/>
        <module name="InterfaceIsType"/>
        <module name="VisibilityModifier"/>

        <!-- Miscellaneous -->
        <module name="ArrayTypeStyle"/>
        <module name="FinalParameters"/>
        <module name="TodoComment"/>
        <module name="UpperEll"/>
    </module>

    <!-- Miscellaneous checks -->
    <module name="RegexpSingleline">
        <property name="format" value="\\s+$"/>
        <property name="minimum" value="0"/>
        <property name="maximum" value="0"/>
        <property name="message" value="Line has trailing spaces."/>
    </module>
</module>
"""
    
    with open(CHECKSTYLE_CONFIG, "w") as f:
        f.write(config_content)

# If this file is run directly, check if Checkstyle is available and download if needed
if __name__ == "__main__":
    if os.path.exists(CHECKSTYLE_PATH):
        print(f"Checkstyle is available at {CHECKSTYLE_PATH}")
    else:
        print("Checkstyle is not available. Attempting to download...")
        if download_checkstyle():
            print("Checkstyle was successfully downloaded.")
        else:
            print("Failed to download Checkstyle. Please download it manually.")