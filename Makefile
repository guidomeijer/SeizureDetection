.ONESHELL:

LINE_LENGTH=99
VENV_NAME?=venv

## ┌───────────────────────────────────────────────────────────────────┐
## │                    Team Wavelet Makefile                          │
## │ ───────────────────────────────────────────────────────────────── │
## │                                                                   │
## └───────────────────────────────────────────────────────────────────┘

.PHONY: setup
setup: ## Initialize the project, create venv and install packages
	virtualenv -p python3.8 venv
	# source venv/bin/activate
	# pip3 install -r requirements.txt

.PHONY: shell
shell: ## Load virtualenv
	. venv/bin/activate

.PHONY: black
black: ## Run Black
	black . --line-length=$(LINE_LENGTH)

.PHONY: isort
isort: ## Run isort
	isort .

.PHONY: flake8
flake8: ## Run Flake8
	flake8 . --exclude=venv --max-line-length=$(LINE_LENGTH)

.PHONY: check
check: black flake8 isort ## Run Black, Flake8, isort

.PHONY: help
help: ## show this help
# regex for general help
	@sed -ne "s/^##\(.*\)/\1/p" $(MAKEFILE_LIST)
# regex for makefile commands (targets)
	@printf "────────────────────────`tput bold``tput setaf 2` Make Commands `tput sgr0`────────────────────────────────\n"
	@sed -ne "/@sed/!s/\(^[^#?=]*:\).*##\(.*\)/`tput setaf 2``tput bold`\1`tput sgr0`\2/p" $(MAKEFILE_LIST)
# regex for makefile variables
	@printf "────────────────────────`tput bold``tput setaf 4` Make Variables `tput sgr0`───────────────────────────────\n"
	@sed -ne "/@sed/!s/\(.*\)?=\(.*\)##\(.*\)/`tput setaf 4``tput bold`\1:`tput setaf 5`\2`tput sgr0`\3/p" $(MAKEFILE_LIST)

# make help the default
.DEFAULT_GOAL := help
