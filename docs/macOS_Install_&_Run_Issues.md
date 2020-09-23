## Obstacles Encountered During Brainlit Tutorial (macOS)
(contact jduva4@jhu.edu for related questions)

1. If using ```virtualenv``` to create the environment, make sure that you have Python 3 installed outside of Anaconda (call ```python --version```) because many systems will not. Make sure that ```pip``` references Python 3 (the ```pip --version``` command should show ```3.xx``` in the path), otherwise ```pip``` installs could be updating Python 2 exclusively. 

2. May run into a schema-related error when importing napari in Step 1: “This is specifically a suppressible warning because if you’re using a schema other than the ones in SUPPORTED_VERSIONS, it’s possible that not all functionality will be supported properly. If you don’t want to see these messages, add a warningfilter to your code.” (Source: https://github.com/cwacek/python-jsonschema-objects/issues/184)

3. If the iPyNb kernel consistently dies when running Step 4, try including ```%gui qt``` just above the ```import napari``` line. 

4. Not exclusive to macOS but make sure aws .json file has no dollar signs in the strings and is being edited/saved within the terminal using a program like Nano or Vim. Do not use external editors like Sublime.
