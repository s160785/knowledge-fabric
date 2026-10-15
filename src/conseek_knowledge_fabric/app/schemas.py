"""Validated HTTP request models for the graph-building API."""

from urllib.parse import urlsplit

from pydantic import BaseModel, Field, SecretStr, field_validator


class BuildRequest(BaseModel):
	repository: str = Field(min_length=1, max_length=2048)
	branch: str | None = Field(default=None, max_length=255)
	neo4j_uri: str = Field(default="bolt://localhost:7687", min_length=1, max_length=512)
	neo4j_username: str = Field(default="neo4j", min_length=1, max_length=256)
	neo4j_password: SecretStr
	neo4j_database: str | None = Field(default=None, max_length=128)
	neo4j_browser_url: str = Field(default="http://localhost:7474/browser/", max_length=2048)

	@field_validator("neo4j_uri")
	@classmethod
	def valid_bolt_uri(cls, value: str) -> str:
		parsed = urlsplit(value)
		if (
			parsed.scheme not in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"}
			or not parsed.hostname
			or parsed.username
			or parsed.password
		):
			raise ValueError("Use a valid Neo4j Bolt URI, such as bolt://localhost:7687")
		return value

	@field_validator("neo4j_browser_url")
	@classmethod
	def valid_browser_url(cls, value: str) -> str:
		parsed = urlsplit(value)
		if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
			raise ValueError("Neo4j Browser URL must be an http(s) URL without embedded credentials")
		return value


class BAChatRequest(BaseModel):
	question: str = Field(min_length=1, max_length=2048)
	neo4j_uri: str = Field(default="bolt://localhost:7687", min_length=1, max_length=512)
	neo4j_username: str = Field(default="neo4j", min_length=1, max_length=256)
	neo4j_password: SecretStr
	neo4j_database: str | None = Field(default=None, max_length=128)

	@field_validator("neo4j_uri")
	@classmethod
	def valid_bolt_uri(cls, value: str) -> str:
		parsed = urlsplit(value)
		if (
			parsed.scheme not in {"bolt", "bolt+s", "bolt+ssc", "neo4j", "neo4j+s", "neo4j+ssc"}
			or not parsed.hostname
			or parsed.username
			or parsed.password
		):
			raise ValueError("Use a valid Neo4j Bolt URI, such as bolt://localhost:7687")
		return value


class BAChatResponse(BaseModel):
	question: str
	answer: str
	evidence: list[dict] = Field(default_factory=list)