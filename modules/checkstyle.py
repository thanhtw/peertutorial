"""
Module for running Checkstyle analysis on Java code.
This module provides functions to check Java code style according to standards.
"""

import subprocess
import os
import re
import xml.etree.ElementTree as ET
import tempfile
import requests
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path


# Path to checkstyle jar file - now based on data directory in module location
CHECKSTYLE_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "checkstyle.jar")
# Path to checkstyle configuration
CHECKSTYLE_CONFIG = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "checkstyle.xml")
# Checkstyle download URL
CHECKSTYLE_DOWNLOAD_URL = "https://github.com/checkstyle/checkstyle/releases/download/checkstyle-10.3.3/checkstyle-10.3.3-all.jar"


def run_checkstyle(java_file_path: str) -> Dict[str, Any]:
    """
    Run Checkstyle on a Java file and return the results.
    
    Args:
        java_file_path: Path to the Java file to check
        
    Returns:
        A dictionary with checkstyle results:
        {
            "success": True/False,
            "issues": [
                {
                    "line": line_number,
                    "message": error_message,
                    "severity": severity_level,
                    "rule": rule_name
                },
                ...
            ],
            "error": "Error message if checkstyle failed to run",
            "command": "Command that was executed"
        }
    """
    result = {
        "success": False,
        "issues": [],
        "error": "",
        "command": ""
    }
    
    try:
        # Ensure the file exists
        if not os.path.exists(java_file_path):
            result["error"] = f"File not found: {java_file_path}"
            return result
        
        # Ensure checkstyle is available
        if not os.path.exists(CHECKSTYLE_PATH):
            if not download_checkstyle():
                result["error"] = "Checkstyle is not available and couldn't be downloaded"
                return result
        
        # Create a temporary file for XML output
        with tempfile.NamedTemporaryFile(suffix=".xml", delete=False) as temp_xml:
            temp_xml_path = temp_xml.name
        
        # Run checkstyle with XML output format
        command = [
            "java", "-jar", CHECKSTYLE_PATH,
            "-c", CHECKSTYLE_CONFIG,
            "-f", "xml",
            "-o", temp_xml_path,
            java_file_path
        ]
        
        result["command"] = " ".join(command)
        
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        # Check if process ran successfully
        if process.returncode != 0 and not os.path.exists(temp_xml_path):
            result["error"] = f"Checkstyle failed: {process.stderr}"
            # Clean up temporary file
            try:
                os.unlink(temp_xml_path)
            except:
                pass
            return result
        
        # Parse the XML output
        try:
            # Check if file exists and has content
            if os.path.exists(temp_xml_path) and os.path.getsize(temp_xml_path) > 0:
                tree = ET.parse(temp_xml_path)
                root = tree.getroot()
                
                for file_elem in root.findall(".//file"):
                    for error_elem in file_elem.findall(".//error"):
                        line = int(error_elem.get("line", 0))
                        message = error_elem.get("message", "Unknown error")
                        severity = error_elem.get("severity", "error")
                        source = error_elem.get("source", "")
                        
                        # Extract rule name from source
                        rule = source.split('.')[-1] if source else "Unknown"
                        
                        result["issues"].append({
                            "line": line,
                            "message": message,
                            "severity": severity,
                            "rule": rule
                        })
                
                result["success"] = True
            else:
                # Empty XML file usually means no style issues
                result["success"] = True
        except ET.ParseError:
            # If XML parsing fails, try to extract information from command output
            result["error"] = "XML parsing failed. Trying to extract from command output."
            if process.stderr:
                for line in process.stderr.split("\n"):
                    match = re.search(r"\[(\d+)\]\s*(.*)", line)
                    if match:
                        line_num = int(match.group(1))
                        message = match.group(2)
                        result["issues"].append({
                            "line": line_num,
                            "message": message,
                            "severity": "error",
                            "rule": "Unknown"
                        })
        
        # Clean up temporary file
        try:
            os.unlink(temp_xml_path)
        except:
            pass
        
        return result
    
    except subprocess.TimeoutExpired:
        result["error"] = "Checkstyle analysis timed out after 30 seconds"
        return result
    
    except FileNotFoundError:
        result["error"] = "Java not found. Please make sure Java is installed."
        return result
    
    except Exception as e:
        result["error"] = f"An error occurred during Checkstyle analysis: {str(e)}"
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
        
        # Check if the file already exists
        if os.path.exists(CHECKSTYLE_PATH):
            return True
            
        print(f"Downloading Checkstyle from {CHECKSTYLE_DOWNLOAD_URL}...")
        
        # Try to download using requests
        try:
            response = requests.get(CHECKSTYLE_DOWNLOAD_URL, stream=True)
            response.raise_for_status()  # Raise an exception for HTTP errors
            
            with open(CHECKSTYLE_PATH, 'wb') as f:
                for chunk in response.iter_content(chunk_size=8192): 
                    if chunk:
                        f.write(chunk)
        except Exception as e:
            print(f"Failed to download using requests: {e}")
            
            # Try using curl as fallback
            try:
                subprocess.run(
                    ["curl", "-L", CHECKSTYLE_DOWNLOAD_URL, "-o", CHECKSTYLE_PATH],
                    check=True,
                    capture_output=True,
                    text=True
                )
            except Exception as curl_e:
                print(f"Failed to download using curl: {curl_e}")
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
    # Check if config already exists
    if os.path.exists(CHECKSTYLE_CONFIG):
        return
        
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
    
    # Create data directory if it doesn't exist
    data_dir = os.path.dirname(CHECKSTYLE_CONFIG)
    os.makedirs(data_dir, exist_ok=True)
    
    with open(CHECKSTYLE_CONFIG, "w") as f:
        f.write(config_content)


def check_checkstyle_available() -> Tuple[bool, str]:
    """
    Check if Checkstyle is available and properly set up.
    
    Returns:
        Tuple of (available, message)
    """
    # Check if JAR file exists
    if not os.path.exists(CHECKSTYLE_PATH):
        if download_checkstyle():
            return True, "Checkstyle was downloaded successfully"
        else:
            return False, f"Checkstyle JAR file not found at {CHECKSTYLE_PATH} and couldn't be downloaded"
    
    # Check if config file exists
    if not os.path.exists(CHECKSTYLE_CONFIG):
        create_default_checkstyle_config()
        if not os.path.exists(CHECKSTYLE_CONFIG):
            return False, f"Checkstyle configuration file not found at {CHECKSTYLE_CONFIG} and couldn't be created"
    
    # Try running a simple test
    try:
        with tempfile.NamedTemporaryFile(suffix=".java", delete=False) as tmp:
            tmp_path = tmp.name
            tmp.write(b"public class Test {}")
            
        command = [
            "java", "-jar", CHECKSTYLE_PATH,
            "-c", CHECKSTYLE_CONFIG,
            tmp_path
        ]
        
        process = subprocess.run(
            command,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        # Clean up temporary file
        try:
            os.unlink(tmp_path)
        except:
            pass
        
        if process.returncode == 0:
            return True, "Checkstyle is working correctly"
        else:
            return False, f"Checkstyle test failed: {process.stderr}"
    
    except Exception as e:
        return False, f"Error testing Checkstyle: {str(e)}"


# If this file is run directly, check if Checkstyle is available and download if needed
if __name__ == "__main__":
    available, message = check_checkstyle_available()
    if available:
        print(f"Checkstyle is available: {message}")
    else:
        print(f"Checkstyle is not available: {message}")