from abc import abstractmethod
from typing import List, Optional

from llama_index.core.schema import NodeWithScore, QueryBundle

class DuplicateRemoverNodePostprocessor:
    """Node postprocessor."""

    @abstractmethod
    def postprocess_nodes(
        self, nodes: List[NodeWithScore], query_bundle: Optional[QueryBundle]
    ) -> List[NodeWithScore]:
        """Postprocess nodes."""
        print("*****_postprocess_nodes enter*****")

        # Prepare all veriables
        unique_hashes = set()
        unique_nodes = []

        # Check every node in provided nodes
        for idx, node in enumerate(nodes):
            print(f"Node {idx}: {node.text}")

            # Get the node hash
            node_hash = node.node.hash

            # Check if node hash was not seen
            if node_hash not in unique_hashes:
                unique_hashes.add(node_hash)
                unique_nodes.append(node)

        return unique_nodes