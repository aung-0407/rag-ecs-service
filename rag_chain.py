import os
import json
import logging
from typing import List, Dict, Any, Optional
import boto3
from opensearchpy import OpenSearch, RequestsHttpConnection, AWSV4SignerAuth
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.schema import HumanMessage, SystemMessage

logger = logging.getLogger(__name__)


class RAGChain:
    """RAG Chain with multi-LLM support (OpenAI, Anthropic, Google)"""
    
    # Separate cache for each provider
    _openai_key_cache = None
    _anthropic_key_cache = None
    _google_key_cache = None
    
    def __init__(self):
        """Initialize RAG chain with OpenSearch and multiple LLM providers"""
        # Get configuration
        self.opensearch_endpoint = os.environ.get("OPENSEARCH_ENDPOINT")
        self.opensearch_index = os.environ.get("OPENSEARCH_INDEX", "documents")
        self.aws_region = os.environ.get("AWS_REGION", "ap-northeast-1")
        
        # Default models for each provider
        self.llm_model = os.environ.get("LLM_MODEL", "gpt-4o-mini")
        self.llm_model_anthropic = os.environ.get("LLM_MODEL_ANTHROPIC", "claude-3-haiku-20240307")
        self.llm_model_google = os.environ.get("LLM_MODEL_GOOGLE", "gemini-2.5-pro")
        
        self.secret_name = os.environ.get("SECRET_NAME", "rag/openai/api_key")
        
        if not self.opensearch_endpoint:
            raise ValueError("OPENSEARCH_ENDPOINT environment variable is required")
        
        logger.info(f"Initializing RAG chain with OpenSearch: {self.opensearch_endpoint}")
        
        # Get API keys (with caching)
        self.openai_api_key = self._get_cached_openai_api_key()
        self.anthropic_api_key = self._get_cached_anthropic_api_key()
        self.google_api_key = self._get_cached_google_api_key()
        
        # Initialize OpenSearch client with auto-refreshing credentials
        self.opensearch_client = self._init_opensearch()
        
        # Initialize S3 client for presigned URLs
        self.s3_client = boto3.client('s3', region_name=self.aws_region)
        
        # Initialize OpenAI embeddings
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=self.openai_api_key
        )
        
        # Initialize LLMs for each provider
        self.llm = ChatOpenAI(
            model=self.llm_model,
            temperature=0.7,
            openai_api_key=self.openai_api_key
        )
        
        self.anthropic_llm = ChatAnthropic(
            model=self.llm_model_anthropic,
            temperature=0.7,
            anthropic_api_key=self.anthropic_api_key
        )
        
        self.google_llm = ChatGoogleGenerativeAI(
            model=self.llm_model_google,
            temperature=0.7,
            google_api_key=self.google_api_key
        )
        
        logger.info("RAG chain initialized successfully with OpenAI, Anthropic, and Google support")
    
    @classmethod
    def _get_cached_openai_api_key(cls) -> str:
        """Retrieve OpenAI API key with caching"""
        if cls._openai_key_cache:
            logger.info("Using cached OpenAI API key")
            return cls._openai_key_cache
        
        # Try environment variable first
        api_key = os.environ.get('OPENAI_API_KEY')
        if api_key:
            logger.info("Using OpenAI API key from environment variable")
            cls._openai_key_cache = api_key
            return api_key
        
        # Fall back to Secrets Manager
        try:
            logger.info("Fetching OpenAI API key from Secrets Manager...")
            secret_name = os.environ.get("SECRET_NAME", "rag/openai/api_key")
            region_name = os.environ.get("AWS_REGION", "ap-northeast-1")
            
            client = boto3.client('secretsmanager', region_name=region_name)
            response = client.get_secret_value(SecretId=secret_name)
            
            secret = json.loads(response['SecretString'])
            api_key = secret.get('RAG_OPENAI_API_KEY')
            
            if not api_key:
                raise ValueError("RAG_OPENAI_API_KEY not found in secret")
            
            cls._openai_key_cache = api_key
            logger.info("Successfully retrieved and cached OpenAI API key")
            return api_key
            
        except Exception as e:
            logger.error(f"Failed to retrieve OpenAI API key: {str(e)}")
            raise
    
    @classmethod
    def _get_cached_anthropic_api_key(cls) -> str:
        """Retrieve Anthropic API key with caching"""
        if cls._anthropic_key_cache:
            logger.info("Using cached Anthropic API key")
            return cls._anthropic_key_cache
        
        # Try environment variable first
        api_key = os.environ.get('ANTHROPIC_API_KEY')
        if api_key:
            logger.info("Using Anthropic API key from environment variable")
            cls._anthropic_key_cache = api_key
            return api_key
        
        # Fall back to Secrets Manager
        try:
            logger.info("Fetching Anthropic API key from Secrets Manager...")
            secret_name = os.environ.get("SECRET_NAME_ANTHROPIC", "rag/anthropic/api_key")
            region_name = os.environ.get("AWS_REGION", "ap-northeast-1")
            
            client = boto3.client('secretsmanager', region_name=region_name)
            response = client.get_secret_value(SecretId=secret_name)
            
            secret = json.loads(response['SecretString'])
            api_key = secret.get('RAG_ANTHROPIC_API_KEY')
            
            if not api_key:
                raise ValueError("RAG_ANTHROPIC_API_KEY not found in secret")
            
            cls._anthropic_key_cache = api_key
            logger.info("Successfully retrieved and cached Anthropic API key")
            return api_key
            
        except Exception as e:
            logger.error(f"Failed to retrieve Anthropic API key: {str(e)}")
            raise
    
    @classmethod
    def _get_cached_google_api_key(cls) -> str:
        """Retrieve Google API key with caching"""
        if cls._google_key_cache:
            logger.info("Using cached Google API key")
            return cls._google_key_cache
        
        # Try environment variable first
        api_key = os.environ.get('GOOGLE_API_KEY')
        if api_key:
            logger.info("Using Google API key from environment variable")
            cls._google_key_cache = api_key
            return api_key
        
        # Fall back to Secrets Manager
        try:
            logger.info("Fetching Google API key from Secrets Manager...")
            secret_name = os.environ.get("SECRET_NAME_GOOGLE", "rag/google/api_key")
            region_name = os.environ.get("AWS_REGION", "ap-northeast-1")
            
            client = boto3.client('secretsmanager', region_name=region_name)
            response = client.get_secret_value(SecretId=secret_name)
            
            secret = json.loads(response['SecretString'])
            api_key = secret.get('RAG_GOOGLE_API_KEY')
            
            if not api_key:
                raise ValueError("RAG_GOOGLE_API_KEY not found in secret")
            
            cls._google_key_cache = api_key
            logger.info("Successfully retrieved and cached Google API key")
            return api_key
            
        except Exception as e:
            logger.error(f"Failed to retrieve Google API key: {str(e)}")
            raise
    
    def _init_opensearch(self) -> OpenSearch:
        """
        Initialize OpenSearch with auto-refreshing credentials
        """
        try:
            session = boto3.Session()
            credentials = session.get_credentials()
            
            auth = AWSV4SignerAuth(credentials, self.aws_region, 'es')
            
            client = OpenSearch(
                hosts=[{'host': self.opensearch_endpoint, 'port': 443}],
                http_auth=auth,
                use_ssl=True,
                verify_certs=True,
                connection_class=RequestsHttpConnection,
                timeout=30,
                max_retries=3,
                retry_on_timeout=True,
                pool_connections=10,
                pool_maxsize=10
            )
            
            info = client.info()
            logger.info(f"Connected to OpenSearch: {info['version']['number']}")
            
            return client
            
        except Exception as e:
            logger.error(f"Failed to initialize OpenSearch: {str(e)}")
            raise
    
    def _generate_presigned_url(self, bucket: str, key: str, expiration: int = 3600) -> str:
        """
        Generate presigned URL for S3 object
        
        Args:
            bucket: S3 bucket name
            key: S3 object key
            expiration: URL expiration time in seconds (default 1 hour)
        
        Returns:
            Presigned URL string
        """
        try:
            url = self.s3_client.generate_presigned_url(
                'get_object',
                Params={'Bucket': bucket, 'Key': key},
                ExpiresIn=expiration
            )
            return url
        except Exception as e:
            logger.error(f"Failed to generate presigned URL for s3://{bucket}/{key}: {e}")
            return None
    
    def test_opensearch_connection(self) -> bool:
        """Test OpenSearch connection"""
        try:
            self.opensearch_client.info()
            return True
        except Exception as e:
            logger.error(f"OpenSearch connection test failed: {e}")
            return False
    
    def search(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "hybrid"
    ) -> List[Dict[str, Any]]:
        """Search documents in OpenSearch"""
        try:
            logger.info(f"Executing {search_type} search for: '{query}' (top_k={top_k})")
            
            if search_type == "text":
                return self._text_search(query, top_k)
            elif search_type == "vector":
                return self._vector_search(query, top_k)
            else:  # hybrid
                return self._hybrid_search(query, top_k)
                
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            raise
    
    def _text_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Full-text search using BM25"""
        search_body = {
            "size": top_k,
            "query": {
                "multi_match": {
                    "query": query,
                    "fields": [
                        "chunk_text^2",
                        "section_path",
                        "filename",
                        "content"
                    ],
                    "type": "best_fields"
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"timestamp": {"order": "desc"}}
            ]
        }
        
        response = self.opensearch_client.search(
            index=self.opensearch_index,
            body=search_body
        )
        
        total_hits = response['hits']['total']['value']
        logger.info(f"Text search found {total_hits} results")
        
        return self._format_search_results(response)
    
    def _vector_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Semantic search using vector embeddings"""
        query_vector = self.embeddings.embed_query(query)
        logger.info(f"Generated query embedding (dimension: {len(query_vector)})")
        
        search_body = {
            "size": top_k,
            "query": {
                "knn": {
                    "embedding": {
                        "vector": query_vector,
                        "k": top_k
                    }
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"timestamp": {"order": "desc"}}
            ]
        }
        
        response = self.opensearch_client.search(
            index=self.opensearch_index,
            body=search_body
        )
        
        total_hits = response['hits']['total']['value']
        logger.info(f"Vector search found {total_hits} results")
        
        return self._format_search_results(response)
    
    def _hybrid_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        """Hybrid search combining text and vector search"""
        query_vector = self.embeddings.embed_query(query)
        logger.info(f"Generated query embedding (dimension: {len(query_vector)})")
        
        search_body = {
            "size": top_k,
            "query": {
                "bool": {
                    "should": [
                        {
                            "multi_match": {
                                "query": query,
                                "fields": [
                                    "chunk_text^2",
                                    "section_path",
                                    "filename",
                                    "content"
                                ],
                                "type": "best_fields",
                                "boost": 1.0
                            }
                        },
                        {
                            "knn": {
                                "embedding": {
                                    "vector": query_vector,
                                    "k": top_k,
                                    "boost": 1.0
                                }
                            }
                        }
                    ],
                    "minimum_should_match": 1
                }
            },
            "sort": [
                {"_score": {"order": "desc"}},
                {"timestamp": {"order": "desc"}}
            ]
        }
        
        response = self.opensearch_client.search(
            index=self.opensearch_index,
            body=search_body
        )
        
        total_hits = response['hits']['total']['value']
        max_score = response['hits']['max_score'] if response['hits']['hits'] else 0
        logger.info(f"Hybrid search found {total_hits} results (max_score: {max_score})")
        
        if response['hits']['hits']:
            top_score = response['hits']['hits'][0]['_score']
            top_doc = response['hits']['hits'][0]['_source'].get('filename', 'Unknown')
            logger.info(f"Top result: {top_doc} (score: {top_score})")
        
        return self._format_search_results(response)
    
    def _format_search_results(self, response: Dict) -> List[Dict[str, Any]]:
        """
        Format OpenSearch results with comprehensive metadata including images
        """
        documents = []
        
        for hit in response['hits']['hits']:
            source = hit['_source']
            
            # Extract basic metadata
            doc = {
                "content": source.get('chunk_text', ''),
                "metadata": {
                    "filename": source.get('filename', 'Unknown'),
                    "document_id": source.get('document_id', 'Unknown'),
                    "section_title": source.get('section_title', ''),
                    "section_path": source.get('section_path', ''),
                    "file_type": source.get('file_type', ''),
                    "bucket": source.get('bucket', ''),
                    "key": source.get('key', ''),
                    "chunk_index": source.get('chunk_index', 0),
                    "total_chunks": source.get('total_chunks', 0),
                    "timestamp": source.get('timestamp', '')
                },
                "score": hit['_score']
            }
            
            # Process image references with presigned URLs
            if 'image_references' in source and source['image_references']:
                images_with_urls = []
                
                for img in source['image_references']:
                    image_data = {
                        'alt_text': img.get('alt_text', ''),
                        'filename': img.get('filename', ''),
                        'section': img.get('section', ''),
                        'section_path': img.get('section_path', ''),
                        'context': img.get('context', ''),
                        'position': img.get('position', 0)
                    }
                    
                    # Generate presigned URL for the image
                    bucket = img.get('s3_bucket')
                    key = img.get('s3_key')
                    
                    if bucket and key:
                        presigned_url = self._generate_presigned_url(bucket, key)
                        if presigned_url:
                            image_data['url'] = presigned_url
                            image_data['s3_bucket'] = bucket
                            image_data['s3_key'] = key
                            logger.info(f"Generated presigned URL for image: {img.get('filename')}")
                    
                    images_with_urls.append(image_data)
                
                doc["metadata"]["images"] = images_with_urls
                doc["metadata"]["image_count"] = len(images_with_urls)
            
            # Include additional metadata if present
            if 'total_images' in source:
                doc["metadata"]["total_images"] = source['total_images']
            if 'total_sections' in source:
                doc["metadata"]["total_sections"] = source['total_sections']
            if 'total_slides' in source:
                doc["metadata"]["total_slides"] = source['total_slides']
            
            documents.append(doc)
        
        return documents
    
    def query(
        self,
        query: str,
        top_k: int = 5,
        search_type: str = "hybrid",
        llm_model: Optional[str] = None
    ) -> Dict[str, Any]:
        """Execute RAG query: search + generate answer"""
        try:
            logger.info(f"Executing RAG query: {query}")
            
            documents = self.search(query, top_k, search_type)
            
            if not documents:
                logger.warning("No documents found for query")
                return {
                    "answer": "I couldn't find any relevant information in the documents to answer your question.",
                    "sources": []
                }
            
            logger.info(f"Retrieved {len(documents)} documents")
            
            # Log if images were found
            total_images = sum(
                len(doc.get('metadata', {}).get('images', [])) 
                for doc in documents
            )
            if total_images > 0:
                logger.info(f"Found {total_images} images in retrieved documents")
            
            answer = self._generate_answer(query, documents, llm_model)
            
            return {
                "answer": answer,
                "sources": documents
            }
            
        except Exception as e:
            logger.error(f"RAG query failed: {str(e)}")
            raise
    
    def _determine_llm_provider(self, model_name: str) -> str:
        """
        Determine which LLM provider to use based on model name
        
        Returns: 'openai', 'anthropic', or 'google'
        """
        model_lower = model_name.lower()
        
        # Anthropic models
        if any(keyword in model_lower for keyword in ['claude', 'anthropic']):
            return 'anthropic'
        
        # Google models
        if any(keyword in model_lower for keyword in ['gemini', 'google', 'palm']):
            return 'google'
        
        # Default to OpenAI for GPT models and unknowns
        return 'openai'
    
    def _generate_answer(
        self, 
        query: str, 
        documents: List[Dict[str, Any]],
        llm_model: Optional[str] = None
    ) -> str:
        """Generate answer using LLM with retrieved context and image information"""
        
        model_to_use = llm_model or self.llm_model
        logger.info(f"Generating answer with model: {model_to_use}")
        
        # Build context
        context_parts = []
        image_references = []
        
        for i, doc in enumerate(documents, 1):
            metadata = doc.get('metadata', {})
            filename = metadata.get('filename', 'Unknown')
            section_path = metadata.get('section_path', '')
            section_title = metadata.get('section_title', '')
            content = doc['content']
            score = doc.get('score', 0)
            
            # Build context with section information
            if section_path:
                context_parts.append(
                    f"[Source {i}: {filename} - {section_path}] (Relevance: {score:.2f})\n{content}\n"
                )
            elif section_title:
                context_parts.append(
                    f"[Source {i}: {filename} - {section_title}] (Relevance: {score:.2f})\n{content}\n"
                )
            else:
                context_parts.append(
                    f"[Source {i}: {filename}] (Relevance: {score:.2f})\n{content}\n"
                )
            
            # Add image information if present
            images = metadata.get('images', [])
            if images:
                context_parts.append(f"\n Related Images in Source {i}:\n")
                for j, img in enumerate(images, 1):
                    img_info = f"  Image {j}: {img.get('filename', 'Unknown')}"
                    if img.get('alt_text'):
                        img_info += f" - {img['alt_text']}"
                    if img.get('section_path'):
                        img_info += f" (in section: {img['section_path']})"
                    context_parts.append(img_info + "\n")
                    
                    # Collect image references for response
                    image_references.append({
                        'source': i,
                        'filename': img.get('filename', 'Unknown'),
                        'url': img.get('url'),
                        'alt_text': img.get('alt_text', ''),
                        'section': img.get('section_path', img.get('section', ''))
                    })
        
        context = "\n".join(context_parts)
        logger.info(f"Context length: {len(context)} characters")
        logger.info(f"Including {len(image_references)} image references")
        
        import re
        is_japanese = bool(re.search(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]', query))
        logger.info(f"Detected language: {'Japanese' if is_japanese else 'English'}")

        # Create language-specific prompts
        if is_japanese:
            system_message = """あなたは提供されたコンテキストに基づいて質問に答える親切なAIアシスタントです。

重要: 必ず日本語で回答してください。絶対に英語を使わないでください。

例:
質問: 次期POS基盤とは何ですか？
回答: 次期POS基盤は、専門店、飲食店、量販店など様々な業態のPOSシステムで共通して利用する機能を備えた基盤システムです。（ソース1より）

指示:
1. 提供されたソースの情報のみを使用して質問に答えてください
2. ソースに十分な情報が含まれていない場合は、明確に述べてください
3. ソース番号を言及してソースを引用してください（例：「ソース1によると...」「ソース2には...」）
4. コンテキストで画像が言及されている場合は、関連する場合に参照してください
5. 簡潔でありながら包括的に答えてください
6. 不確かな場合は、不確実性を認めてください

重要: 提供されたソースからの情報のみを使用してください。外部の知識を追加しないでください。"""

            user_message = f"""ドキュメントからのコンテキスト:
{context}

質問: {query}

上記のコンテキストに基づいて、日本語で詳細な回答を提供してください。必ず日本語で答えてください。"""

        else:
            system_message = """You are a helpful AI assistant that answers questions based on the provided context.

Example:
Question: What is the next-generation POS platform?
Answer: The next-generation POS platform is a base system with features commonly used by POS systems for various business types including specialty stores, restaurants, and mass retailers. (Source 1)

Instructions:
1. Answer the question using ONLY the information from the provided sources
2. If the sources don't contain enough information, say so clearly
3. Cite your sources by mentioning the source number (e.g., "According to Source 1..." "Source 2 states...")
4. If images are mentioned in the context, reference them when relevant
5. Be concise but comprehensive
6. If you're not sure, acknowledge the uncertainty

Remember: Only use information from the provided sources. Do not add external knowledge."""

            user_message = f"""Context from documents:
{context}

Question: {query}

Please provide a detailed answer based on the context above in English."""

        messages = [
            SystemMessage(content=system_message),
            HumanMessage(content=user_message)
        ]
        
        # Determine which LLM to use and invoke
        provider = self._determine_llm_provider(model_to_use)
        logger.info(f"Using LLM provider: {provider}")
        
        if provider == 'anthropic':
            # Use Anthropic with the specific model
            llm = ChatAnthropic(
                model=model_to_use,
                temperature=0.7,
                anthropic_api_key=self.anthropic_api_key
            )
            response = llm.invoke(messages)
        elif provider == 'google':
            # Use Google with the specific model
            llm = ChatGoogleGenerativeAI(
                model=model_to_use,
                temperature=0.7,
                google_api_key=self.google_api_key
            )
            response = llm.invoke(messages)
        else:
            # Use OpenAI with the specific model
            llm = ChatOpenAI(
                model=model_to_use,
                temperature=0.7,
                openai_api_key=self.openai_api_key
            )
            response = llm.invoke(messages)
        
        logger.info(f"Generated answer ({len(response.content)} characters)")
        
        return response.content