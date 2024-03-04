# Code Execution

TaskWeaver is a code-first agent framework, which means that it always converts the user request into code 
and executes the code to generate the response. In our current implementation, we use a Jupyter Kernel
to execute the code. We choose Jupyter Kernel because it is a well-established tool for interactive computing,
and it supports many programming languages.

## Two Modes of Code Execution

TaskWeaver supports two modes of code execution: `local` and `container`. 
The `local` mode is the default mode. The key difference between the two modes is that the `container` mode
executes the code inside a Docker container, which provides a more secure environment for code execution, while
the `local` mode executes the code as a subprocess of the TaskWeaver process.
As a result, in the `local` mode, if the user has malicious intent, the user could potentially
instruct TaskWeaver to execute harmful code on the host machine. In addition, the LLM could also generate
harmful code, leading to potential security risks.

>💡We recommend using the `container` mode for code execution, especially when the usage of the agent
is open to untrusted users. In the `container` mode, the code is executed in a Docker container, which is isolated
from the host machine. 

## How to Configure the Code Execution Mode

To configure the code execution mode, you need to set the `execution_service.kernel_mode` parameter in the
`taskweaver_config.json` file. The value of the parameter could be `local` or `container`. The default value
is `local`.

TaskWeaver supports the `local` mode without any additional setup. However, to use the `container` mode,
there are a few prerequisites:

- Docker is installed on the host machine.
- A Docker image is built and available on the host machine for code execution.
- The `execution_service.kernel_mode` parameter is set to `container` in the `taskweaver_config.json` file.

Once the code repository is cloned to your local machine, you can build the Docker image
by running the following command in the root directory of the code repository:

```bash
cd scripts

# based on your OS
./build.ps1 # for Windows
./build.sh # for Linux or macOS
```

After the Docker image is built, you can run `docker images` to check if a Docker image 
named `executor_container` is available. 
If the prerequisite is met, you can now run TaskWeaver in the `container` mode.

After running TaskWeaver in the `container` mode, you can check if the container is running by running `docker ps`.
You should see a container of image `taskweaver/executor` running after executing some code. 

## Limitations of the `container` Mode

The `container` mode is more secure than the `local` mode, but it also has some limitations:

- The startup time of the `container` mode is longer than the `local` mode, because it needs to start a Docker container. 
- As the Jupyter Kernel is running inside a Docker container, it has limited access to the host machine. We are mapping the
  `project/workspace/sessions/<session_id>` directory to the container, so the code executed in the container can access the
  files in it. One implication of this is that the user cannot ask the agent to load a file from the host machine, because the
  file is not available in the container. Instead, the user needs to upload the file either using the `/upload` command in 
  the console or the `upload` button in the web interface.
- We have installed required packages in the Docker image to run the Jupyter Kernel. If the user needs to use a package that is
  not available in the Docker image, the user needs to add the package to the Dockerfile (at `TaskWeaver/ces_container/Dockerfile`) 
  and rebuild the Docker image.

