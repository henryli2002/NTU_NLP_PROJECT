"""
基础功能测试脚本
用于快速测试各个模块是否正常工作
"""
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.embedding_extractor import EmbeddingExtractor


def test_embedding_extraction():
    """测试embedding提取"""
    print("Testing embedding extraction...")
    
    try:
        # 测试BERT模型（相对较小）
        extractor = EmbeddingExtractor("bert-base-uncased")
        
        # 测试单个文本
        text = "Hello, world!"
        embedding = extractor.encode(text)
        print(f"  ✓ Single text embedding shape: {embedding.shape}")
        
        # 测试多个文本
        texts = ["Hello", "World", "Test"]
        embeddings = extractor.encode(texts)
        print(f"  ✓ Multiple texts embedding shape: {embeddings.shape}")
        
        # 测试相似度计算
        sim = extractor.get_similarity("I love this", "I really like this")
        print(f"  ✓ Similarity calculation: {sim:.4f}")
        
        print("  ✓ Embedding extraction test passed!\n")
        return True
        
    except Exception as e:
        print(f"  ✗ Embedding extraction test failed: {e}\n")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*50)
    print("Running Basic Tests")
    print("="*50)
    
    test_embedding_extraction()
    
    print("="*50)
    print("Basic tests completed!")
    print("="*50)

