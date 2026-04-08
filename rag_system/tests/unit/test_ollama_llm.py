# rag_system/tests/unit/test_ollama_llm.py
"""Testes unitários do OllamaLLM.

Verifica o cliente Ollama local sem servidor real. O módulo ollama e as
settings são mockados em cada teste. Cobre disponibilidade (is_available
via ollama.list), geração (generate) e propagação de exceções de timeout
ou conexão recusada.
"""
import pytest
from unittest.mock import patch
from rag_system.retrieval.generator.ollama_llm import OllamaLLM


class TestOllamaLLM:
    """Testa o OllamaLLM em isolamento — sem servidor Ollama real.

    is_available verifica se ollama.list() responde sem exceção.
    generate deve retornar a string da chave 'response' do ollama.generate(),
    com strip aplicado. Exceções de rede devem ser propagadas sem swallow.
    """

    def test_instantiation(self):
        assert OllamaLLM() is not None

    @patch("rag_system.retrieval.generator.ollama_llm.ollama")
    def test_is_available_true_when_server_up(self, mock_ollama):
        """ollama.list() sem exceção indica servidor disponível."""
        mock_ollama.list.return_value = {"models": []}
        assert OllamaLLM().is_available() is True

    @patch("rag_system.retrieval.generator.ollama_llm.ollama")
    def test_is_available_false_when_server_down(self, mock_ollama):
        """ollama.list() lançando exceção indica servidor indisponível."""
        mock_ollama.list.side_effect = Exception("Connection refused")
        assert OllamaLLM().is_available() is False

    @patch("rag_system.retrieval.generator.ollama_llm.settings")
    @patch("rag_system.retrieval.generator.ollama_llm.ollama")
    def test_generate_returns_string(self, mock_ollama, mock_settings):
        mock_settings.ollama_model = "llama3.2:3b"
        mock_settings.llm_temperature = 0.1
        mock_ollama.generate.return_value = {"response": "  Resposta do Ollama  "}

        result = OllamaLLM().generate("Quem é Ramon?")
        assert isinstance(result, str)
        assert "Resposta do Ollama" in result

    @patch("rag_system.retrieval.generator.ollama_llm.settings")
    @patch("rag_system.retrieval.generator.ollama_llm.ollama")
    def test_generate_propagates_exception(self, mock_ollama, mock_settings):
        """Timeouts e erros de rede devem ser propagados sem swallow."""
        mock_settings.ollama_model = "llama3.2:3b"
        mock_settings.llm_temperature = 0.1
        mock_ollama.generate.side_effect = Exception("Timeout")

        with pytest.raises(Exception, match="Timeout"):
            OllamaLLM().generate("pergunta")