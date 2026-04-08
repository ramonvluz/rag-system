# rag_system/tests/unit/test_groq_llm.py
"""Testes unitários do GroqLLM.

Verifica a integração com a API Groq sem realizar chamadas reais de rede.
O cliente Groq e as settings são mockados em cada teste para garantir
isolamento total. Cobre disponibilidade (is_available), geração (generate)
e propagação correta de exceções da API.
"""
import pytest
from unittest.mock import patch, MagicMock
from rag_system.retrieval.generator.groq_llm import GroqLLM


class TestGroqLLM:
    """Testa o GroqLLM em isolamento — sem chamadas reais à API Groq.

    is_available depende exclusivamente da presença de GROQ_API_KEY nas
    settings. generate deve repassar a resposta do cliente e propagar
    qualquer exceção sem engolir o erro original.
    """

    @patch("rag_system.retrieval.generator.groq_llm.settings")
    @patch("rag_system.retrieval.generator.groq_llm.Groq")
    def test_instantiation(self, mock_groq_class, mock_settings):
        mock_settings.groq_api_key = "test-key"
        assert GroqLLM() is not None

    @patch("rag_system.retrieval.generator.groq_llm.settings")
    @patch("rag_system.retrieval.generator.groq_llm.Groq")
    def test_is_available_true_when_key_set(self, mock_groq_class, mock_settings):
        mock_settings.groq_api_key = "valid-api-key"
        assert GroqLLM().is_available() is True

    @patch("rag_system.retrieval.generator.groq_llm.settings")
    @patch("rag_system.retrieval.generator.groq_llm.Groq")
    def test_is_available_false_when_key_empty(self, mock_groq_class, mock_settings):
        """Chave vazia deve tornar o LLM indisponível para a lógica de fallback."""
        mock_settings.groq_api_key = ""
        assert GroqLLM().is_available() is False

    @patch("rag_system.retrieval.generator.groq_llm.settings")
    @patch("rag_system.retrieval.generator.groq_llm.Groq")
    def test_generate_returns_string(self, mock_groq_class, mock_settings):
        mock_settings.groq_api_key = "test-key"
        mock_settings.groq_model = "llama-3.1-8b-instant"
        mock_settings.llm_temperature = 0.1

        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="  Resposta do Groq  "))]
        )

        result = GroqLLM().generate("Qual a experiência de Ramon?")
        assert isinstance(result, str)
        assert "Resposta do Groq" in result

    @patch("rag_system.retrieval.generator.groq_llm.settings")
    @patch("rag_system.retrieval.generator.groq_llm.Groq")
    def test_generate_calls_api_once(self, mock_groq_class, mock_settings):
        """A API do Groq deve ser chamada exatamente uma vez por generate()."""
        mock_settings.groq_api_key = "test-key"
        mock_settings.groq_model = "llama-3.1-8b-instant"
        mock_settings.llm_temperature = 0.1

        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        mock_client.chat.completions.create.return_value = MagicMock(
            choices=[MagicMock(message=MagicMock(content="ok"))]
        )

        GroqLLM().generate("pergunta")
        mock_client.chat.completions.create.assert_called_once()

    @patch("rag_system.retrieval.generator.groq_llm.settings")
    @patch("rag_system.retrieval.generator.groq_llm.Groq")
    def test_generate_propagates_exception(self, mock_groq_class, mock_settings):
        """Exceções da API Groq devem ser propagadas sem swallow."""
        mock_settings.groq_api_key = "test-key"
        mock_settings.groq_model = "llama-3.1-8b-instant"
        mock_settings.llm_temperature = 0.1

        mock_client = MagicMock()
        mock_groq_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            GroqLLM().generate("pergunta")