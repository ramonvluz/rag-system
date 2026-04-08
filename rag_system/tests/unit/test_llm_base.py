# rag_system/tests/unit/test_llm_base.py
"""Testes unitários do LLMBase (ABC).

Verifica o contrato da classe base abstrata via implementações concretas
definidas localmente. Não testa GroqLLM nem OllamaLLM — apenas garante
que o protocolo da interface é respeitado e que NotImplementedError é
lançado quando os métodos abstratos são chamados sem implementação.
"""
import pytest
from rag_system.retrieval.generator.base import LLMBase


class ConcreteLLM(LLMBase):
    """Implementação mínima válida de LLMBase — usada para testes positivos."""

    def generate(self, prompt: str) -> str:
        return f"resposta: {prompt}"

    def is_available(self) -> bool:
        return True


class UnavailableLLM(LLMBase):
    """Implementação que simula um LLM inacessível — usada para testes negativos."""

    def generate(self, prompt: str) -> str:
        raise RuntimeError("LLM indisponível")

    def is_available(self) -> bool:
        return False


class TestLLMBase:
    """Testa os contratos da ABC LLMBase via implementações concretas locais.

    Verifica que subclasses válidas funcionam corretamente e que chamar os
    métodos abstratos diretamente na classe base lança NotImplementedError,
    garantindo que o contrato da interface seja sempre respeitado.
    """

    def test_concrete_llm_instantiates(self):
        assert ConcreteLLM() is not None

    def test_generate_returns_string(self):
        result = ConcreteLLM().generate("Quem é Ramon?")
        assert isinstance(result, str) and len(result) > 0

    def test_is_available_returns_bool(self):
        assert isinstance(ConcreteLLM().is_available(), bool)

    def test_is_available_true(self):
        assert ConcreteLLM().is_available() is True

    def test_is_available_false(self):
        assert UnavailableLLM().is_available() is False

    def test_generate_raises_when_unavailable(self):
        with pytest.raises(RuntimeError):
            UnavailableLLM().generate("teste")

    def test_base_generate_raises_not_implemented(self):
        """Chamar generate() direto na base, sem override, deve lançar NotImplementedError."""
        with pytest.raises(NotImplementedError):
            LLMBase.generate(ConcreteLLM(), "teste")

    def test_base_is_available_raises_not_implemented(self):
        """Chamar is_available() direto na base, sem override, deve lançar NotImplementedError."""
        with pytest.raises(NotImplementedError):
            LLMBase.is_available(ConcreteLLM())