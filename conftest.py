import pytest

def pytest_addoption(parser):
    parser.addoption("--get_url", action="store", default='test')

@pytest.fixture
def get_url(request):
    return request.config.getoption("--get_url")
