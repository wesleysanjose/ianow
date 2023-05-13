import os
from langchain.agents import AgentType
from langchain.agents import initialize_agent
from langchain.agents.agent_toolkits.jira.toolkit import JiraToolkit
from langchain.llms import OpenAI
from langchain.utilities.jira import JiraAPIWrapper
from langchain.chains.question_answering import load_qa_chain
from transformers import pipeline
from langchain.llms.huggingface_pipeline import HuggingFacePipeline

from unittest.mock import patch

from jira import JIRA
from atlassian import Jira

import os

from doc_search_llm.servers.arg_parser import load_parser
from doc_search_llm.modules.model_processor import ModelProcessor
from utils.simple_logger import Log
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.absolute()))
log = Log.get_logger(__name__)


class JiraProcessor:

    # Constructor
    def __init__(self):
        self.jira_url = os.environ.get('JIRA_INSTANCE_URL')
        log.debug(f'Initializing JiraProcessor with {self.jira_url}')

        self.jira_username = os.environ.get('JIRA_USERNAME')
        # print first first 4 chars of jira_username
        log.debug(f'jira_username: {self.jira_username[:4]}')

        self.jira_password = os.environ.get('JIRA_PASSWORD')
        # print last 4 chars of password
        log.debug(f'jira_password: {self.jira_password[-4:]}')

        self.api_token = os.environ.get('JIRA_API_TOKEN')
        # print last 4 chars of api token
        log.debug(f'api_token: {self.api_token[-4:]}')

        self.jira_client = None

    def token(self):
        try:
            self.jira_client = JIRA(
                server=self.jira_url, username=self.jira_username, password=self.api_token)
            log.info(f'Connected to Jira at {self.jira_url}')
            return self.jira_client
        except Exception as e:
            log.error(f'Error connecting to Jira: {e}')
            raise e

    # Connect to Jira
    def connect(self):
        try:
            self.jira_client = JIRA(server=self.jira_url, basic_auth=(
                self.jira_username, self.jira_password))
            log.info(f'Connected to Jira at {self.jira_url}')
            return self.jira_client
        except Exception as e:
            log.error(f'Error connecting to Jira: {e}')
            raise e

    def create_issue(self):
        issue_dict = {
            'project': {'key': 'DEV'},
            'summary': 'New issue from jira-python',
            'description': 'Look into this one',
            'issuetype': {'name': 'Task'},
        }
        try:
            new_issue = self.jira_client.create_issue(fields=issue_dict)
            log.info(f'Created new issue: {new_issue}')
        except Exception as e:
            log.error(f'Error creating issue: {e}')
            raise e
        return new_issue

    def get_issue(self, issue_id):
        try:
            issue = self.jira_client.issue(issue_id)
            log.info(f'Got issue: {issue}')
        except Exception as e:
            log.error(f'Error getting issue: {e}')
            raise e
        return issue

    def list_projects(self):
        try:
            projects = self.jira_client.projects()
            log.info(f'Got projects: {projects}')
        except Exception as e:
            log.error(f'Error getting projects: {e}')
            raise e
        return projects

    def list_issues(self, project_id):
        try:
            issues = self.jira_client.search_issues(f'project={project_id}')
            log.info(f'Got issues: {issues}')
        except Exception as e:
            log.error(f'Error getting issues: {e}')
            raise e
        return issues


def simple_test():
    jira = Jira(
        url=os.environ.get('JIRA_INSTANCE_URL'),
        username=os.environ.get('JIRA_USERNAME'),
        password='')
    JQL = 'project = DEV AND issuetype = Task'
    data = jira.jql(JQL)
    print(data)

def langchain_test(args):
    # load the LLM model
    try:
        jira = JiraAPIWrapper()
        toolkit = JiraToolkit.from_jira_api_wrapper(jira)
        log.info(f'Loaded toolkit: {toolkit}')

        model, tokenizer = ModelProcessor.load_model(args)

        # # create the LLM pipeline
        pipe = pipeline("text-generation", model=model,
                        tokenizer=tokenizer, max_new_tokens=1024)
        llm = HuggingFacePipeline(pipeline=pipe)

        # load the QA chain
        chain = load_qa_chain(llm, chain_type="stuff")
        agent = initialize_agent(
            toolkit.get_tools(),
            llm,
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True
        )

        # patch the langchain jira cloud login by using jira server login
        from typing import Dict

        def new_validate_environment(cls, values: Dict) -> Dict:
            """Validate that api key and python package exists in environment."""
            log.debug(f'hijacked successfully')
            jira_processor = JiraProcessor()
            jira = jira_processor.connect()
            values["jira"] = jira

            return values

        # with patch.object(JiraAPIWrapper, 'validate_environment', new=new_validate_environment):
        agent.run(
            "get all the issues from project devops")
    except Exception as e:
        log.error(f'Error loading model: {e}')
        raise e


if __name__ == '__main__':
    args = load_parser()
    langchain_test(args)
    # simple_test()
