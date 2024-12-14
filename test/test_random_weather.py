from eprllib.Utils.random_weather import get_random_weather
from unittest.mock import patch
import numpy as np
import os
import pytest
import tempfile

class TestRandomWeather:

    def test_get_random_weather_2(self):
        """
        Test that get_random_weather returns a valid EPW file path when the folder contains EPW files.
        """
        # Mock the os.listdir function to return a list with EPW files
        mock_files = ['file1.txt', 'weather1.epw', 'weather2.epw', 'file2.doc']
        mock_folder_path = '/mock/folder/path'

        with patch('os.listdir', return_value=mock_files):
            with patch('numpy.random.randint', side_effect=[0, 1]):  # First return non-EPW, then EPW file index
                result = get_random_weather(mock_folder_path)

        # Check that the result is a valid path
        assert isinstance(result, str)
        assert result.endswith('.epw')
        assert os.path.basename(result) in mock_files
        assert os.path.dirname(result) == mock_folder_path

    def test_get_random_weather_3(self):
        """
        Test that get_random_weather selects a random .epw file when non-epw files are present
        """
        with patch('os.listdir') as mock_listdir, \
             patch('numpy.random.randint') as mock_randint, \
             patch('os.path.join') as mock_join:

            # Mock folder content with non-epw files and one epw file
            mock_listdir.return_value = ['file1.txt', 'file2.csv', 'weather.epw']
            
            # Mock random selection to first choose a non-epw file, then the epw file
            mock_randint.side_effect = [0, 2]
            
            # Mock os.path.join to return a predictable path
            mock_join.return_value = '/path/to/weather.epw'

            # Call the function
            result = get_random_weather('/mock/folder/path')

            # Assert that the function returns the expected path
            assert result == '/path/to/weather.epw'

            # Verify that numpy.random.randint was called twice
            assert mock_randint.call_count == 2

            # Verify that os.path.join was called with the correct arguments
            mock_join.assert_called_once_with('/mock/folder/path', 'weather.epw')

    def test_get_random_weather_empty_folder(self):
        """
        Test get_random_weather with an empty folder.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            with pytest.raises(ValueError, match="The folder does not contain any .epw files."):
                get_random_weather(temp_dir)

    def test_get_random_weather_empty_string(self):
        """
        Test get_random_weather with an empty string as input.
        """
        with pytest.raises(FileNotFoundError):
            get_random_weather("")

    def test_get_random_weather_incorrect_type(self):
        """
        Test get_random_weather with incorrect input type.
        """
        with pytest.raises(TypeError):
            get_random_weather(123)

    def test_get_random_weather_invalid_path(self):
        """
        Test get_random_weather with an invalid folder path.
        """
        with pytest.raises(FileNotFoundError):
            get_random_weather("/non/existent/path")

    def test_get_random_weather_no_epw_files(self):
        """
        Test get_random_weather with a folder containing non-epw files.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some non-epw files
            open(os.path.join(temp_dir, "file1.txt"), "w").close()
            open(os.path.join(temp_dir, "file2.csv"), "w").close()

            with pytest.raises(ValueError, match="The folder does not contain any .epw files."):
                get_random_weather(temp_dir)

    def test_get_random_weather_permission_denied(self):
        """
        Test get_random_weather with a folder that has no read permissions.
        """
        with tempfile.TemporaryDirectory() as temp_dir:
            os.chmod(temp_dir, 0o000)  # Remove all permissions
            with pytest.raises(PermissionError):
                get_random_weather(temp_dir)
            os.chmod(temp_dir, 0o754)

    def test_get_random_weather_raises_error_for_empty_folder(self):
        """
        Test that get_random_weather raises a ValueError when the folder does not contain any .epw files.
        """
        # Create a temporary directory
        with pytest.raises(ValueError) as excinfo:
            get_random_weather('nonexistent_folder')
        assert str(excinfo.value) == "The folder does not contain any .epw files."

    def test_get_random_weather_returns_valid_epw_file(self, tmp_path):
        """
        Test that get_random_weather returns a valid .epw file path when the folder contains .epw files.
        """
        # Create a temporary directory with some .epw files
        epw_folder = tmp_path / "epw_files"
        epw_folder.mkdir()
        (epw_folder / "file1.epw").touch()
        (epw_folder / "file2.epw").touch()
        (epw_folder / "file3.txt").touch()

        # Mock random choice to always return the first file
        np.random.randint = lambda start, end: 0

        result = get_random_weather(str(epw_folder))

        assert result.endswith('.epw')
        assert os.path.exists(result)
        assert os.path.basename(result) in os.listdir(str(epw_folder))

    def test_get_random_weather_returns_valid_epw_file_2(self):
        """
        Test that get_random_weather returns a valid .epw file path when the folder contains both .epw and non-.epw files.
        """
        with patch('os.listdir') as mock_listdir, \
             patch('numpy.random.randint') as mock_randint:
            
            mock_listdir.return_value = ['file1.txt', 'weather.epw', 'file2.csv', 'climate.epw']
            mock_randint.side_effect = [0, 1]  # First return non-epw file, then epw file
            
            epw_files_folder_path = '/path/to/epw/files'
            result = get_random_weather(epw_files_folder_path)
            
            assert result.endswith('.epw')
            assert os.path.basename(result) in ['weather.epw', 'climate.epw']
            assert os.path.dirname(result) == epw_files_folder_path