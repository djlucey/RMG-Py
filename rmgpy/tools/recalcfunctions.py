###############################################################################
#                                                                             #
# RMG - Reaction Mechanism Generator                                          #
#                                                                             #
# Copyright (c) 2002-2023 Prof. William H. Green (whgreen@mit.edu),           #
# Prof. Richard H. West (r.west@neu.edu) and the RMG Team (rmg_py@mit.edu)   #
#                                                                             #
# Permission is hereby granted, free of charge, to any person obtaining a    #
# copy of this software and associated documentation files (the 'Software'),  #
# to deal in the Software without restriction, including without limitation   #
# the rights to use, copy, modify, merge, publish, distribute, sublicense,   #
# and/or sell copies of the Software, and to permit persons to whom the      #
# Software is furnished to do so, subject to the following conditions:        #
#                                                                             #
# The above copyright notice and this permission notice shall be included in  #
# all copies or substantial portions of the Software.                         #
#                                                                             #
# THE SOFTWARE IS PROVIDED 'AS IS', WITHOUT WARRANTY OF ANY KIND, EXPRESS OR  #
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,    #
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE #
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER      #
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING     #
# FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER         #
# DEALINGS IN THE SOFTWARE.                                                   #
#                                                                             #
###############################################################################

"""
Utility functions for estimating surface species thermo from adjacency lists.

Thermo is estimated fresh via RMG thermo libraries + Pt111 adsorption group
corrections.  Optional plus/times adjustments (keyed by adsorption group node
name) are applied on top, exactly as in a standard RMG run.

Example usage::

    from rmgpy.tools.recalcfunctions import recalculate_species_thermo

    recalc_config = {
        'thermo_libraries': ['primaryThermoLibrary', 'surfaceThermo'],
        'metal_to_scale_to': 'Ru0001',  # optional; None keeps Pt111 values
        # OR an explicit BE dict:
        # 'metal_to_scale_to': {'C': (-6.750, 'eV/molecule'), 'H': (-2.479, 'eV/molecule'),
        #                       'O': (-3.950, 'eV/molecule'), 'N': (-4.100, 'eV/molecule')},
        'plus_adjust':  {'R': 500.0},   # additive correction (J/mol) for node 'R'
        'times_adjust': {'R': 1.05},    # multiplicative correction for node 'R'
        'species': ['CO*', 'OH*'],      # optional subset; None => all surface species
    }

    adj_dict = {
        'CO*': \"\"\"
    1 C u0 p0 c0 {2,D} {3,D}
    2 O u0 p2 c0 {1,D}
    3 X u0 p0 c0 {1,D}
    \"\"\",
        'OH*': \"\"\"...\"\"\",
    }

    # input_type='adjlist' (default) — pass adjacency list strings directly
    updated_species = recalculate_species_thermo(adj_dict, recalc_config)

    # input_type='species' — pass pre-built Species objects
    updated_species = recalculate_species_thermo(species_dict, recalc_config,
                                                 input_type='species')
"""

import logging
import os

from rmgpy import settings
from rmgpy.data.thermo import ThermoDatabase
from rmgpy.thermo import ThermoData, Wilhoit
from rmgpy.species import Species


def build_species_dict(adj_dict):
    """
    Build a ``{label: Species}`` dict from adjacency list strings.

    Parameters
    ----------
    adj_dict : dict
        ``{label: adjacency_list_string}``

    Returns
    -------
    dict
        ``{label: Species}`` with resonance structures generated.
    """
    result = {}
    for label, adjlist in adj_dict.items():
        spc = Species(label=label).from_adjacency_list(adjlist)
        spc.generate_resonance_structures()
        result[label] = spc
    return result


def recalculate_species_thermo(species_input, recalc_config, db_path=None, input_type='adjlist'):
    """
    Estimate thermo for surface species from scratch using RMG thermo libraries
    and Pt111 adsorption group corrections.

    Thermo is always re-estimated from the database — no incoming thermo is used.
    Plus/times adjustments are applied automatically inside ``get_thermo_data``
    via the adsorption group tree, identical to a standard RMG run.

    Parameters
    ----------
    species_input : dict
        Input depends on *input_type*:

        ``'adjlist'`` (default)
            ``{label: adjacency_list_string}``
        ``'species'``
            ``{label: Species}`` — any pre-assigned ``.thermo`` is ignored;
            thermo is always re-estimated.

    recalc_config : dict
        Keys:

        thermo_libraries : list of str, optional
            Thermo library names to load (e.g. ``['primaryThermoLibrary',
            'surfaceThermo']``).  If omitted, all available libraries are loaded.
        metal_to_scale_to : str or dict, optional
            Target metal for linear scaling from Pt111.  Either a metal label
            from the RMG surface database (e.g. ``'Ru0001'``) or an explicit
            binding energy dict::

                {'C': (-6.750, 'eV/molecule'), 'H': (-2.479, 'eV/molecule'),
                 'O': (-3.950, 'eV/molecule'), 'N': (-4.100, 'eV/molecule')}

            When omitted (or ``None``), thermo is returned on Pt111 with no
            further scaling.
        plus_adjust : dict, optional
            ``{adsorption_group_node_name: float}`` additive H298 corrections
            (J/mol) applied after the adsorption correction.
        times_adjust : dict, optional
            ``{adsorption_group_node_name: float}`` multiplicative corrections
            applied to the adsorption binding energy shift.
        species : list of str, optional
            Labels to process.  ``None`` (default) processes all surface species.

    db_path : str, optional
        Path to the ``RMG-database/input`` directory.  Overrides
        ``settings['database.directory']`` when provided.
    input_type : {'adjlist', 'species'}
        How to interpret *species_input* values.

    Returns
    -------
    list of Species
        Species objects with freshly estimated ``.thermo``.  Species whose
        thermo estimation fails are skipped with a warning.
    """
    if input_type not in ('adjlist', 'species'):
        raise ValueError(f"input_type must be 'adjlist' or 'species', got {input_type!r}")

    if input_type == 'adjlist':
        species_dict = build_species_dict(species_input)
    else:
        species_dict = dict(species_input)

    if db_path is not None:
        settings['database.directory'] = db_path

    thermo_libraries = recalc_config.get('thermo_libraries', None)
    metal_to_scale_to = recalc_config.get('metal_to_scale_to', None)
    plus_adjust = recalc_config.get('plus_adjust', {})
    times_adjust = recalc_config.get('times_adjust', {})
    species_labels = recalc_config.get('species', None)

    # Determine which species to process
    if species_labels is not None:
        missing = [l for l in species_labels if l not in species_dict]
        if missing:
            logging.warning('Labels not found in input and will be skipped: %s', missing)
        to_process = {l: species_dict[l] for l in species_labels if l in species_dict}
    else:
        to_process = {l: s for l, s in species_dict.items() if s.contains_surface_site()}

    if not to_process:
        logging.warning('recalculate_species_thermo: no surface species found to process.')
        return []

    thermo_db = ThermoDatabase()
    thermo_db.load(
        path=os.path.join(settings['database.directory'], 'thermo'),
        libraries=thermo_libraries,
        depository=False,
        surface=True,
    )
    thermo_db.plus_adjust = plus_adjust
    thermo_db.times_adjust = times_adjust

    # Resolve the target metal for get_thermo_data.
    # If an explicit BE dict is given, load it onto thermo_db.binding_energies and
    # pass metal_to_scale_to=None so correct_binding_energy uses self.binding_energies.
    if isinstance(metal_to_scale_to, dict):
        thermo_db.set_binding_energies(metal_to_scale_to)
        scale_to_arg = None
    else:
        scale_to_arg = metal_to_scale_to  # string name or None (stays at Pt111)

    updated = []
    for label, spc in to_process.items():
        spc.thermo = None  # discard any pre-existing thermo; always re-estimate
        try:
            spc.thermo = thermo_db.get_thermo_data(spc, metal_to_scale_to=scale_to_arg)
        except Exception:
            logging.warning('Thermo estimation failed for %s; skipping.', label, exc_info=True)
            continue

        # Convert ThermoData/Wilhoit to NASA so downstream species.to_cantera() works.
        # This mirrors thermoengine.process_thermo_data and ensures P_ref=10000 Pa
        # (hardcoded in NASA.to_cantera) is used, consistent with the cantera1 writer.
        if isinstance(spc.thermo, (ThermoData, Wilhoit)):
            spc.thermo = spc.thermo.to_nasa(Tmin=100.0, Tmax=5000.0, Tint=1000.0)

        logging.debug('Estimated thermo for %s: H298 = %.1f kJ/mol',
                      label, spc.thermo.get_enthalpy(298.0) / 1e3)
        updated.append(spc)

    return updated
